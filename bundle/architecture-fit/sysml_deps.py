"""
SysML-native transition dependency extractor.

Builds the per-variable next-state dependency graph DIRECTLY from the SysML model
(via sysml_parser, the same parser the shield uses) -- NOT from the exported SMV,
which lossily drops the controller->state-machine-actuator wire.

get_model(path) -> dict(STATE, ACTIONS, nsupp, copies, OBS, R), consumed by
reconstruct_closure. Canonical names drop the root part name read from SysML
and replace `::` with `_`.

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
                          AcceptStmt, IfStmt, AssignStmt, PerformStmt, InParamStmt)


def _refs(e):
    out = []
    def w(x):
        if isinstance(x, RefExpr): out.append(tuple(x.path))
        elif isinstance(x, BinaryExpr): w(x.left); w(x.right)
        elif isinstance(x, UnaryExpr): w(x.operand)
        elif isinstance(x, TernaryExpr): w(x.condition); w(x.true_expr); w(x.false_expr)
    if e is not None: w(e)
    return out


def _canon(parts, root=None):
    parts = [p for p in parts if p]
    if root and parts and parts[0] == root:
        parts = parts[1:]
    return "_".join(parts)


class SysMLModel:
    def __init__(self, path):
        p = SysMLParser(path); p.parse()
        self.p = p
        if not p.system_part:
            raise ValueError("SysML file must contain exactly one root part instance")
        self.sys = p.system_part
        self.inst_by_name = {v.name: v for v in p.part_instances.values()}
        self.insts = set(self.inst_by_name.keys())
        # action_name -> body, for `perform action X` inlining (Modbus models put
        # the coil-write sends in separate performed actions).
        self.actions_by_name = {a.name: a.body
                                for pd in p.part_defs.values() for a in pd.actions}

        # ---- neural action def: actions, the controller instance, in-bindings ----
        (
            self.neural_out,
            self.in_binds,
            self.ctrl_inst,
            self.neural_call,
        ) = self._find_neural()
        self.ctrl_fqn = f"{self.sys}::{self.ctrl_inst}" if self.ctrl_inst else self.sys
        self.ACTIONS = {
            self._canon([self.ctrl_inst, self.neural_call, output])
            for output in self.neural_out
        }

        # ---- state-variable targets (known before resolving deps) ----
        self.state_targets = set()
        for sa in p.step_actions:
            self.state_targets.add(self._canon(sa.target_key.split("::")))
        for fqn, sm in p.instance_state_machines.items():
            inst = fqn.split("::")[-1]
            self.state_targets.add(f"{inst}_state")
            for t in sm.transitions:
                for d in (t.do_action or []):
                    if isinstance(d, AssignStmt):
                        self.state_targets.add(self._canon([inst] + d.target))

        # ---- numeric constants (exclude anything that's actually a state var) ----
        self.consts = set()
        for prm in p.parameters:
            self.consts.add(self._canon(prm.qualified_name.split("::")))
        for inst in p.part_instances.values():
            for a in inst.attributes:
                self.consts.add(self._canon([inst.name, a]))
        for fqn, statements in p.step_action_bodies:
            ctx = fqn.split("::")
            for statement in statements:
                if isinstance(statement, InParamStmt):
                    self.consts.add(self._qual([statement.name], ctx))
        self.consts -= self.state_targets

        # ---- same-cycle alias map: canonical-prefix -> canonical-prefix ----
        self.alias = {}
        for f in p.flows:                                   # receiver <- sender
            self._alias(self._pc(f.to_port), self._pc(f.from_port))
        for frm, to in p.connects:                          # receiver <- sender
            self._alias(self._pc(to), self._pc(frm))
        for lhs, rhs in p.parsed_bindings.items():          # bind lhs = rhs
            self._alias(self._canon(lhs.split("::")), self._canon(rhs.split("::")))
        self._wire_send_accept()                            # ports carry sent items

        # derived attrs (DEFINEs / `attribute x = expr`) are SAME-cycle computed
        # values -> expanded to their dependencies during dep collection.
        self.derived = {}
        for d in p.derived_attributes:
            ctx = d.context.split("::") if d.context else [self.sys]
            self.derived[self._canon(d.qualified_name.split("::"))] = (d.expression, ctx)
        self._constraints()

        # ---- build deps ----
        self.STATE = set()
        self.nsupp = {}
        self.copies = set()
        self._build_steps()
        self._build_state_machines()
        self.OBS = self._build_obs()
        self.R = self._build_R()

    # ---------- low-level ----------
    def _canon(self, parts):
        return _canon(parts, self.sys)

    def _alias(self, a, b):
        if a and b and a != b:
            self.alias[a] = b

    def _pc(self, dotted):
        return self._canon(dotted.split("."))

    def _qual(self, parts, ctx):
        """Qualify a context-local ref to a rooted canonical key (no aliasing)."""
        if parts[0] in self.insts or parts[0] == self.sys:
            return self._canon(parts)
        return self._canon(list(ctx) + list(parts))

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
        """Turn SysML equalities and implications into same-cycle edges."""
        for c in self.p.parsed_constraints:
            if "ScenarioConstraint" in getattr(c, "metadata", []):
                continue
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
                    send_index[self._canon([inst] + s.port.split("."))] = self._canon([inst, s.item_name])
                    self._alias(self._port_key(inst, s.port), self._canon([inst, s.item_name]))
        # accepts: link the accepted var DIRECTLY to the message sent on the
        # connected port's matching sub-channel (handles bidirectional Modbus
        # req/resp); fall back to the local port for simple unidirectional ports.
        def wire_accept(inst, var, port):
            parts = port.split(".")
            connected = cmap.get(self._canon([inst, parts[0]]))
            lookup = self._canon([connected] + parts[1:]) if connected else None
            if lookup and lookup in send_index:
                self._alias(self._canon([inst, var]), send_index[lookup])
            else:
                self._alias(self._canon([inst, var]), self._port_key(inst, port))
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
                    self._alias(self._canon([inst] + s.target),
                                self._canon([inst] + list(s.expr.path)))

    def _port_key(self, inst, port):
        """Canonical key for a port ref. Bare port -> append its single item slot;
        dotted port ('thermometerPort.reading') -> use as given."""
        parts = port.split(".")
        if len(parts) == 1:
            slot = self._port_slot(inst, port)
            if slot:
                parts = parts + [slot]
        return self._canon([inst] + parts)

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
        if self.p.controller_part is None:
            raise ValueError(
                "SysML model must contain exactly one part instance that owns a #Neural action"
            )
        ctrl_fqn = self.p.controller_part
        ctrl_instance = self.p.part_instances[ctrl_fqn]
        part_def = self.p.part_defs[ctrl_instance.part_type]
        neural_defs = [
            action for action in part_def.action_defs if "Neural" in action.metadata
        ]
        if len(neural_defs) != 1:
            raise ValueError("the #Neural action owner must define exactly one #Neural action")
        neural = neural_defs[0]
        calls_by_identity = {
            id(statement): statement
            for action in part_def.actions
            for statement in self._flatten(action.body)
            if isinstance(statement, SubactionCallStmt)
            and statement.type_name == neural.name
        }
        calls = list(calls_by_identity.values())
        if len(calls) != 1:
            raise ValueError(
                f"expected exactly one call to #Neural action {neural.name}, found {len(calls)}"
            )
        call = calls[0]
        binds = {
            binding.name: tuple(binding.expr.path)
            for binding in call.bindings
            if isinstance(binding, InputBindingStmt)
            and isinstance(binding.expr, RefExpr)
        }
        self.is_continuous = any(
            (getattr(output, "type_name", "") or "").lower() not in {"bool", "boolean"}
            for output in neural.out_params
        )
        return (
            [output.name for output in neural.out_params],
            binds,
            ctrl_instance.name,
            call.name,
        )

    # ---------- dependency edges ----------
    def _keep(self, k):
        return k in self.ACTIONS or k in self.state_targets or \
               k not in self.consts

    def _build_steps(self):
        for sa in self.p.step_actions:
            tgt = self._canon(sa.target_key.split("::"))
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
            latch = self._latch_action(sm, inst)
            state_changing = any(t.from_state != t.to_state for t in sm.transitions)
            trig_vars = {t.trigger_var for t in sm.transitions if t.trigger_var}
            trig_port = next((t.trigger_port for t in sm.transitions if t.trigger_port), None)
            if latch and state_changing:                    # the SM state IS the latch
                self.STATE.add(sv); self.nsupp[sv] = {latch}; self.copies.add(sv)
            for t in sm.transitions:
                for d in (t.do_action or []):
                    if not isinstance(d, AssignStmt):
                        continue
                    tk = self._canon([inst] + d.target)
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
        for a condition on the extracted neural-action call whose body sends
        through that port."""
        if not trig_port:
            return None
        ctrl_port = cmap.get(self._canon([inst, trig_port.split(".")[0]]))
        if not ctrl_port:
            return None
        field = self._find_decision_for_port(ctrl_port)
        return self._canon([self.ctrl_inst, self.neural_call, field]) if field else None

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
                    if r and r[0] == self.neural_call:
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
                if self._canon([self.ctrl_inst] + s.port.split(".")).startswith(ctrl_port):
                    return cur
        return None

    def _latch_action(self, sm, inst):
        ports = {t.trigger_port for t in sm.transitions if t.trigger_port}
        for port in ports:
            src = self.alias.get(self._canon([inst, port]))
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
                    return self._canon([self.ctrl_inst, self.neural_call, hit])
        return None

    def _scan_if_send(self, stmts, ctrl_port_canon):
        for s in stmts:
            if isinstance(s, IfStmt):
                for snd in [d for d in s.body if isinstance(d, SendStmt)]:
                    if self._canon([self.ctrl_inst, snd.port]) == ctrl_port_canon:
                        for r in _refs(s.condition):
                            if r and r[0] == self.neural_call:
                                return r[-1]
                r = (self._scan_if_send(s.body, ctrl_port_canon)
                     or self._scan_if_send(s.else_body, ctrl_port_canon))
                if r: return r
        return None

    def _build_obs(self):
        obs = set()
        ctx = self.ctrl_fqn.split("::")
        for nm, ref in self.in_binds.items():
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

def get_model(path):
    m = SysMLModel(path)
    return dict(STATE=m.STATE, ACTIONS=m.ACTIONS, nsupp=m.nsupp,
                copies=m.copies, OBS=m.OBS, R=m.R,
                continuous=getattr(m, "is_continuous", False))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("provide one SysML file path")
    path = sys.argv[1]
    m = SysMLModel(path)
    print(f"MODEL: {path}")
    print(f"ACTIONS: {sorted(m.ACTIONS)}")
    print(f"OBS:     {sorted(m.OBS)}")
    print(f"R:       {sorted(m.R)}")
    print("STATE next-deps:")
    for v in sorted(m.STATE):
        c = " [copy]" if v in m.copies else ""
        print(f"   {v}{c}  <-  {sorted(m.nsupp.get(v, set()))}")
