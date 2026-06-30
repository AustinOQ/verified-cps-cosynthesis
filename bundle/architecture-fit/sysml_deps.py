"""
SysML-native transition dependency extractor.

Builds the per-variable next-state dependency graph DIRECTLY from the SysML model
(via sysml_parser, the same parser the shield uses) -- NOT from the exported SMV,
which lossily drops the controller->state-machine-actuator wire.

get_model(path) -> dict(STATE, ACTIONS, nsupp, copies, OBS, R), consumed by
reconstruct_closure. Canonical names match the SMV (drop 'system', '::'->'_')
so the graph can be cross-checked against the SMV for the parts it gets right.

Edges:
  step_actions / state-machines  -> NEXT-cycle dep
  flows / connects / binds / derived / send-accept -> SAME-cycle alias (value eq)
"""
import os, sys
_SM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sysml-models")
if _SM not in sys.path:
    sys.path.insert(0, _SM)
from sysml_parser import (SysMLParser, RefExpr, BinaryExpr, UnaryExpr, TernaryExpr,
                          LiteralExpr, SubactionCallStmt, InputBindingStmt, SendStmt,
                          AcceptStmt, IfStmt, AssignStmt, PerformStmt)


def _refs(e):
    out = []
    def w(x):
        if isinstance(x, RefExpr): out.append(tuple(x.path))
        elif isinstance(x, BinaryExpr): w(x.left); w(x.right)
        elif isinstance(x, UnaryExpr): w(x.operand)
        elif isinstance(x, TernaryExpr): w(x.condition); w(x.true_expr); w(x.false_expr)
    if e is not None: w(e)
    return out


def _canon(parts):
    parts = [p for p in parts if p]
    if parts and parts[0] == "system":
        parts = parts[1:]
    return "_".join(parts)


class SysMLModel:
    def __init__(self, path):
        p = SysMLParser(path); p.parse()
        self.p = p
        self.sys = p.system_part or "system"
        self.inst_by_name = {v.name: v for v in p.part_instances.values()}
        self.insts = set(self.inst_by_name.keys())
        # action_name -> body, for `perform action X` inlining (Modbus models put
        # the coil-write sends in separate performed actions).
        self.actions_by_name = {a.name: a.body
                                for pd in p.part_defs.values() for a in pd.actions}

        # ---- neural action def: actions, the controller instance, in-bindings ----
        self.neural_out, self.in_binds, self.ctrl_inst = self._find_neural()
        self.ctrl_fqn = f"{self.sys}::{self.ctrl_inst}" if self.ctrl_inst else self.sys
        self.ACTIONS = {_canon([self.ctrl_inst, "policyCall", o]) for o in self.neural_out}

        # ---- state-variable targets (known before resolving deps) ----
        self.state_targets = set()
        for sa in p.step_actions:
            self.state_targets.add(_canon(sa.target_key.split("::")))
        for fqn, sm in p.instance_state_machines.items():
            inst = fqn.split("::")[-1]
            self.state_targets.add(f"{inst}_state")
            for t in sm.transitions:
                for d in (t.do_action or []):
                    if isinstance(d, AssignStmt):
                        self.state_targets.add(_canon([inst] + d.target))

        # ---- numeric constants (exclude anything that's actually a state var) ----
        self.consts = {"dt"}
        for prm in p.parameters:
            self.consts.add(_canon(prm.qualified_name.split("::")))
        for inst in p.part_instances.values():
            for a in inst.attributes:
                self.consts.add(_canon([inst.name, a]))
        self.consts -= self.state_targets

        # ---- same-cycle alias map: canonical-prefix -> canonical-prefix ----
        self.alias = {}
        for f in p.flows:                                   # receiver <- sender
            self._alias(self._pc(f.to_port), self._pc(f.from_port))
        for frm, to in p.connects:                          # receiver <- sender
            self._alias(self._pc(to), self._pc(frm))
        for lhs, rhs in p.parsed_bindings.items():          # bind lhs = rhs
            self._alias(_canon(lhs.split("::")), _canon(rhs.split("::")))
        self._wire_send_accept()                            # ports carry sent items

        # derived attrs (DEFINEs / `attribute x = expr`) are SAME-cycle computed
        # values -> expanded to their dependencies during dep collection.
        self.derived = {}
        for d in p.derived_attributes:
            ctx = d.context.split("::") if d.context else [self.sys]
            self.derived[_canon(d.qualified_name.split("::"))] = (d.expression, ctx)
        self._constraints()      # flow constraints: flowRate <- isRunning/isOpen, etc.

        # ---- build deps ----
        self.STATE = set()
        self.nsupp = {}
        self.copies = set()
        self._build_steps()
        self._build_state_machines()
        self.OBS = self._build_obs()
        self.R = self._build_R()

    # ---------- low-level ----------
    def _alias(self, a, b):
        if a and b and a != b:
            self.alias[a] = b

    def _pc(self, dotted):
        return _canon(dotted.split("."))

    def _qual(self, parts, ctx):
        """Qualify a context-local ref to a rooted canonical key (no aliasing)."""
        if parts[0] in self.insts or parts[0] == self.sys:
            return _canon(parts)
        return _canon(list(ctx) + list(parts))

    def _deref(self, key):
        for _ in range(30):
            hit = False
            for pre in sorted(self.alias, key=len, reverse=True):
                if key == pre or key.startswith(pre + "_"):
                    key = self.alias[pre] + key[len(pre):]
                    hit = True; break
            if not hit:
                return key
        return key

    def _resolve(self, parts, ctx):
        return self._deref(self._qual(parts, ctx))

    def _collect_refs(self, expr, ctx):
        deps = set()
        for r in _refs(expr):
            deps |= self._collect_one(list(r), ctx)
        return deps

    def _collect_one(self, ref, ctx, seen=None):
        return self._expand_key(self._resolve(ref, ctx), seen or set())

    def _expand_key(self, k, seen):
        if k in self.derived and k not in seen:               # same-cycle DEFINE
            seen = seen | {k}
            expr, dctx = self.derived[k]
            out = set()
            for r in _refs(expr):
                out |= self._collect_one(list(r), dctx, seen)
            return out
        return {k} if self._keep(k) else set()

    def _conjuncts(self, e):
        if isinstance(e, BinaryExpr) and e.op == "and":
            return self._conjuncts(e.left) + self._conjuncts(e.right)
        return [e]

    def _constraints(self):
        """Turn flow constraints into same-cycle derived edges:
          `isRunning implies outlet.flowRateMl == maxFlow`  -> outlet.flow <- isRunning
          `feederTank.outlet.flow == valve.outlet.flow`     -> tank.outlet <- valve.outlet
        Added to self.derived so dependency collection expands them transitively."""
        for c in self.p.parsed_constraints:
            ctx = c.context.split("::") if c.context else [self.sys]
            for conj in self._conjuncts(c.expression):
                if isinstance(conj, BinaryExpr) and conj.op == "implies":
                    A, B = conj.left, conj.right
                    if (isinstance(B, BinaryExpr) and B.op == "==" and
                            isinstance(B.left, RefExpr)):
                        key = self._qual(B.left.path, ctx)   # the controlled flow port
                        # depends on the condition (isRunning) AND the assigned value
                        self.derived.setdefault(key, (BinaryExpr("and", A, B.right), ctx))
                elif (isinstance(conj, BinaryExpr) and conj.op == "==" and
                        isinstance(conj.left, RefExpr)):
                    key = self._qual(conj.left.path, ctx)
                    self.derived.setdefault(key, (conj.right, ctx))

    def _part_ctx(self, key_parts):
        """Longest prefix ending in a part instance (the owning part)."""
        last = 0
        for i, tok in enumerate(key_parts):
            if tok in self.insts:
                last = i
        return key_parts[:last + 1]

    # ---------- send/accept wiring ----------
    def _wire_send_accept(self):
        bodies = []
        loop_bodies = []   # SM self-loops: same-cycle Modbus computations
        for fqn, stmts in self.p.step_action_bodies:
            bodies.append((fqn.split("::")[-1], stmts))
        for name, pd in self.p.part_defs.items():
            insts = [i.name for i in self.p.part_instances.values() if i.part_type == name]
            for act in pd.actions:
                for inst in insts:
                    bodies.append((inst, act.body))
        sm_triggers = []
        for fqn, sm in self.p.instance_state_machines.items():
            inst = fqn.split("::")[-1]
            for t in sm.transitions:
                if t.trigger_var and t.trigger_port:
                    sm_triggers.append((inst, t.trigger_var, t.trigger_port))
                if t.do_action and t.from_state == t.to_state:
                    loop_bodies.append((inst, t.do_action))
        allb = bodies + loop_bodies
        cmap = {}
        for frm, to in self.p.connects:
            a, b = self._pc(frm), self._pc(to); cmap[a] = b; cmap[b] = a
        # index every send by its port (+ sub-channel) -> the sent message
        send_index = {}
        for inst, stmts in allb:
            for s in self._flatten(stmts):
                if isinstance(s, SendStmt):
                    send_index[_canon([inst] + s.port.split("."))] = _canon([inst, s.item_name])
                    self._alias(self._port_key(inst, s.port), _canon([inst, s.item_name]))
        # accepts: link the accepted var DIRECTLY to the message sent on the
        # connected port's matching sub-channel (handles bidirectional Modbus
        # req/resp); fall back to the local port for simple unidirectional ports.
        def wire_accept(inst, var, port):
            parts = port.split(".")
            connected = cmap.get(_canon([inst, parts[0]]))
            lookup = _canon([connected] + parts[1:]) if connected else None
            if lookup and lookup in send_index:
                self._alias(_canon([inst, var]), send_index[lookup])
            else:
                self._alias(_canon([inst, var]), self._port_key(inst, port))
        for inst, stmts in allb:
            for s in self._flatten(stmts):
                if isinstance(s, AcceptStmt) and s.var_name:
                    wire_accept(inst, s.var_name, s.port)
        for inst, var, port in sm_triggers:
            wire_accept(inst, var, port)
        # same-cycle item-field copies in SM self-loops (sensor response read)
        for inst, stmts in loop_bodies:
            for s in self._flatten(stmts):
                if (isinstance(s, AssignStmt) and len(s.target) >= 2
                        and isinstance(s.expr, RefExpr)):
                    self._alias(_canon([inst] + s.target),
                                _canon([inst] + list(s.expr.path)))

    def _port_key(self, inst, port):
        """Canonical key for a port ref. Bare port -> append its single item slot;
        dotted port ('thermometerPort.reading') -> use as given."""
        parts = port.split(".")
        if len(parts) == 1:
            slot = self._port_slot(inst, port)
            if slot:
                parts = parts + [slot]
        return _canon([inst] + parts)

    def _port_slot(self, inst, port):
        pt = self.inst_by_name.get(inst)
        if not pt: return None
        ptype = self.p.part_def_ports.get(pt.part_type, {}).get(port)
        items = self.p.port_def_items.get(ptype, {}) if ptype else {}
        return next(iter(items)) if len(items) == 1 else None

    def _flatten(self, stmts):
        for s in stmts:
            if isinstance(s, PerformStmt) and s.action_name in self.actions_by_name:
                yield from self._flatten(self.actions_by_name[s.action_name])
                continue
            yield s
            if isinstance(s, IfStmt):
                yield from self._flatten(s.body); yield from self._flatten(s.else_body)

    # ---------- neural ----------
    def _find_neural(self):
        for name, pd in self.p.part_defs.items():
            for ad in pd.action_defs:
                if "Neural" in ad.metadata:
                    self.is_continuous = any(
                        getattr(o, "type_name", None) == "Real" for o in ad.out_params)
                    ctrl = next((i.name for i in self.p.part_instances.values()
                                 if i.part_type == name), None)
                    binds = {}
                    for act in pd.actions:
                        for st in self._flatten(act.body):
                            if isinstance(st, SubactionCallStmt) and st.type_name == ad.name:
                                for b in st.bindings:
                                    if isinstance(b, InputBindingStmt) and isinstance(b.expr, RefExpr):
                                        binds[b.name] = tuple(b.expr.path)
                    return [o.name for o in ad.out_params], binds, ctrl
        return [], {}, None

    # ---------- dependency edges ----------
    def _keep(self, k):
        if k == "dt" or k.endswith("_dt"):
            return False
        if ("flowrate" in k.lower() and k not in self.derived
                and k not in self.state_targets):
            return False   # unconnected/underived flow port -> 0
        return k in self.ACTIONS or k in self.state_targets or \
               (k not in self.consts and not self._is_const_name(k))

    def _build_steps(self):
        for sa in self.p.step_actions:
            tgt = _canon(sa.target_key.split("::"))
            ctx = self._part_ctx(sa.target_key.split("::"))
            self.STATE.add(tgt)
            deps = self._collect_refs(sa.expression, ctx)
            self.nsupp[tgt] = deps
            if isinstance(sa.expression, RefExpr) and len(deps) == 1:
                self.copies.add(tgt)

    def _build_state_machines(self):
        cmap = {}                                # port connect map, both directions
        for frm, to in self.p.connects:
            a, b = self._pc(frm), self._pc(to)
            cmap[a] = b; cmap[b] = a
        for fqn, sm in self.p.instance_state_machines.items():
            inst = fqn.split("::")[-1]
            sv = f"{inst}_state"
            latch = self._latch_action(sm, inst)            # on/reset coupling (thermostat)
            state_changing = any(t.from_state != t.to_state for t in sm.transitions)
            trig_vars = {t.trigger_var for t in sm.transitions if t.trigger_var}
            trig_port = next((t.trigger_port for t in sm.transitions if t.trigger_port), None)
            if latch and state_changing:                    # the SM state IS the latch
                self.STATE.add(sv); self.nsupp[sv] = {latch}; self.copies.add(sv)
            for t in sm.transitions:
                for d in (t.do_action or []):
                    if not isinstance(d, AssignStmt):
                        continue
                    tk = _canon([inst] + d.target)
                    if any(r and r[0] in trig_vars for r in _refs(d.expr)):
                        # coil-write actuator latch: attr := triggermsg.field
                        # -> couple to the controller's decision (collapses to last cmd)
                        dec = self._actuator_decision(inst, trig_port, cmap)
                        self.STATE.add(tk)
                        self.nsupp[tk] = {dec} if dec else {tk}
                        self.copies.add(tk)
                    elif latch and state_changing:
                        # latch output following the SM state (e.g. heatOut := watts)
                        self.STATE.add(tk); self.nsupp[tk] = {sv}; self.copies.add(tk)
                    # else: self-loop item-field copy -> alias (in _wire_send_accept)

    def _actuator_decision(self, inst, trig_port, cmap):
        """For a coil-write actuator, find the controller decision that drives it:
        the actuator's cmd port connects to a controller port; scan the controller
        for `if policyCall.X { ... send <coil> via <that port> }` -> X."""
        if not trig_port:
            return None
        ctrl_port = cmap.get(_canon([inst, trig_port.split(".")[0]]))
        if not ctrl_port:
            return None
        field = self._find_decision_for_port(ctrl_port)
        return _canon([self.ctrl_inst, "policyCall", field]) if field else None

    def _find_decision_for_port(self, ctrl_port):
        for pd in self.p.part_defs.values():
            if any(i.part_type == pd.name and i.name == self.ctrl_inst
                   for i in self.p.part_instances.values()):
                for act in pd.actions:
                    r = self._scan_dec(act.body, ctrl_port, None)
                    if r:
                        return r
        return None

    def _scan_dec(self, stmts, ctrl_port, cur):
        for s in stmts:
            if isinstance(s, IfStmt):
                d = cur
                for r in _refs(s.condition):
                    if r and r[0] == "policyCall":
                        d = r[-1]
                res = self._scan_dec(s.body, ctrl_port, d) \
                    or self._scan_dec(s.else_body, ctrl_port, cur)
                if res:
                    return res
            elif isinstance(s, PerformStmt) and s.action_name in self.actions_by_name:
                res = self._scan_dec(self.actions_by_name[s.action_name], ctrl_port, cur)
                if res:
                    return res
            elif isinstance(s, SendStmt) and cur:
                if _canon([self.ctrl_inst] + s.port.split(".")).startswith(ctrl_port):
                    return cur
        return None

    def _latch_action(self, sm, inst):
        ports = {t.trigger_port for t in sm.transitions if t.trigger_port}
        for port in ports:
            src = self.alias.get(_canon([inst, port]))   # e.g. controller_heaterControlPort
            if not src:
                continue
            a = self._controlport_action(src)
            if a:
                return a
        return None

    def _controlport_action(self, ctrl_port_canon):
        for name, pd in self.p.part_defs.items():
            for act in pd.actions:
                hit = self._scan_if_send(act.body, ctrl_port_canon)
                if hit:
                    return _canon([self.ctrl_inst, "policyCall", hit])
        return None

    def _scan_if_send(self, stmts, ctrl_port_canon):
        for s in stmts:
            if isinstance(s, IfStmt):
                for snd in [d for d in s.body if isinstance(d, SendStmt)]:
                    if _canon([self.ctrl_inst, snd.port]) == ctrl_port_canon:
                        for r in _refs(s.condition):
                            if r and r[0] == "policyCall":
                                return r[-1]
                r = (self._scan_if_send(s.body, ctrl_port_canon)
                     or self._scan_if_send(s.else_body, ctrl_port_canon))
                if r: return r
        return None

    def _build_obs(self):
        obs = set()
        ctx = self.ctrl_fqn.split("::")
        for nm, ref in self.in_binds.items():
            if nm.lower() == "done":
                continue
            for k in self._collect_one(list(ref), ctx):
                if k in self.state_targets:
                    obs.add(k)
        return obs & self.STATE

    def _build_R(self):
        R = set()
        for req in self.p.parsed_requirements:
            ctx = req.context.split("::") if req.context else [self.sys]
            for r in _refs(req.expression):
                rr = list(r)
                if rr and rr[0] == req.subject_var:
                    rr = rr[1:]
                if not rr:
                    continue
                for k in self._collect_one(rr, ctx):
                    if k in self.STATE:
                        R.add(k)
        return R

    def _is_const_name(self, k):
        return any(s in k.lower() for s in
                   ("setpoint", "targetspeed", "tolerance", "safefollow", "outside",
                    "coefficient", "masskg", "volume", "maxforce", "maxengine",
                    "maxbrake", "onthreshold", "leadspeed", "original", "transfer",
                    "gradepercent", "propagationdelay", "heatoutputwatts"))


def get_model(path):
    m = SysMLModel(path)
    return dict(STATE=m.STATE, ACTIONS=m.ACTIONS, nsupp=m.nsupp,
                copies=m.copies, OBS=m.OBS, R=m.R,
                continuous=getattr(m, "is_continuous", False))


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_SM, "thermostat", "model.sysml")
    m = SysMLModel(path)
    print(f"MODEL: {path}")
    print(f"ACTIONS: {sorted(m.ACTIONS)}")
    print(f"OBS:     {sorted(m.OBS)}")
    print(f"R:       {sorted(m.R)}")
    print("STATE next-deps:")
    for v in sorted(m.STATE):
        c = " [copy]" if v in m.copies else ""
        print(f"   {v}{c}  <-  {sorted(m.nsupp.get(v, set()))}")
