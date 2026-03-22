// generate-combined-invariants
package TankFillingSystem {

    // Standard library imports
    private import ScalarValues::*;

    // Modbus message hierarchy.
    item def ModbusMsg;
    item def ModbusReqMsg :> ModbusMsg;
    item def ModbusResMsg :> ModbusMsg;

    // Neural annotations.
    metadata def Neural;

    metadata def NeuralRequirement;

    // Prohibition / Obligation annotations.
    metadata def Prohibition;

    metadata def Obligation;

    metadata def Termination;

    // Scenario input annotations.
    metadata def ScenarioInput;

    metadata def ScenarioConstraint;

    item def ReadInputRegisterReqMsg :> ModbusReqMsg {
        attribute registerAddress : Integer;
    }

    item def WriteCoilReqMsg :> ModbusReqMsg {
        attribute coilAddress : Integer;
        attribute coilValue : Boolean;
    }

    item def ReadInputRegisterRespMsg :> ModbusResMsg {
        attribute response : Integer;
    }

    item def ExceptionResMsg :> ModbusResMsg;

    // A Modbus message port.
    port def ModbusMasterPort {
        out item req : ModbusReqMsg;
        in item resp : ModbusResMsg;
    }

    item def VolumeReading {
        attribute volumeMl : Integer;
    }

    // A port to connect the volume sensor to a tank.
    port def VolumeSensorToTankPort {
        in item reading : VolumeReading;
    }

    // Represents a Modbus client.
    part def VolumeSensor {
        // Connection to Modbus master.
        port modbusPort : ~ModbusMasterPort;

        // Connection to tank.
        port tankConnection : VolumeSensorToTankPort;

        state def VolumeSensorBehavior {
            entry; then waiting;

            state waiting;
            transition waiting_to_waiting
                first waiting
                accept msg : ReadInputRegisterReqMsg via modbusPort.req
                if msg.registerAddress == 0 
                do action {
                    item rsp : ReadInputRegisterRespMsg;
                    assign rsp.response := tankConnection.reading.volumeMl;
                    send rsp via modbusPort.resp;
                }
                then waiting;


            transition wait_to_waiting_invalid_msg
            first waiting
            accept msg : ModbusReqMsg via modbusPort.req
            do action {
                item rsp : ExceptionResMsg;
                send rsp via modbusPort.resp;
            }
            then waiting;
        }

        exhibit state behavior : VolumeSensorBehavior;
    }

    // Item that flows through the system
    item def Fluid;

    // Port definition with flow rate
    port def FluidPort {
        attribute flowRateMl : Integer;
    }

    // Tank with level tracking
    part def Tank {
        port outlet : FluidPort;
        port inlet : FluidPort;

        port volumeSensorPort : ~VolumeSensorToTankPort;

        attribute capacityMl : Integer;
        attribute currentLevelMl : Integer;
        attribute isFull : Boolean = currentLevelMl >= capacityMl;
        attribute isEmpty : Boolean = currentLevelMl <= 0;

        // Timestep update: advance currentLevel by net flow * dt
        action step {
            in dt : Real;
            assign currentLevelMl := currentLevelMl + (inlet.flowRateMl - outlet.flowRateMl) * dt;
        }

        bind volumeSensorPort.reading.volumeMl = currentLevelMl;
    }

    // Pump with on/off state and flow rate
    part def Pump {
        port inlet : FluidPort;
        port outlet : FluidPort;

        port modbusPort : ~ModbusMasterPort;

        attribute isRunning : Boolean;
        attribute maxFlowRateMl : Integer;

        state def PumpBehavior {
            entry; then waiting;

            state waiting;
            transition waiting_to_waiting
                first waiting 
                accept coilWriteRequest : WriteCoilReqMsg via modbusPort.req
                if coilWriteRequest.coilAddress == 0
                do action { 
                    assign isRunning := coilWriteRequest.coilValue; 
                    item res : ModbusResMsg;
                    send res via modbusPort.resp;
                }
                then waiting;

            transition waiting_to_waiting_err
                first waiting 
                accept req : ModbusReqMsg via modbusPort.req
                do action {
                    item res : ExceptionResMsg;
                    send res via modbusPort.resp;
                }
                then waiting;
        }

        exhibit state behavior : PumpBehavior;

        // When running, outlet flow equals max rate; otherwise zero
        constraint flowConstraint {
            (isRunning implies outlet.flowRateMl == maxFlowRateMl) and
            (not isRunning implies outlet.flowRateMl == 0)
        }
    }

    // Valve with position control
    part def Valve {
        port inlet : FluidPort;
        port outlet : FluidPort;

        attribute isOpen : Boolean;

        port modbusPort : ~ModbusMasterPort;

        state def ValveBehavior {
            entry; then waiting;

            state waiting;
            transition waiting_to_waiting
                first waiting 
                accept coilWriteRequest : WriteCoilReqMsg via modbusPort.req
                if coilWriteRequest.coilAddress == 0
                do action { 
                    assign isOpen := coilWriteRequest.coilValue; 
                    item res : ModbusResMsg;
                    send res via modbusPort.resp;
                }
                then waiting;

            transition waiting_to_waiting_err
                first waiting 
                accept req : ModbusReqMsg via modbusPort.req
                do action {
                    item res : ExceptionResMsg;
                    send res via modbusPort.resp;
                }
                then waiting;
        }

        exhibit state behavior : ValveBehavior;

        // Outlet flow is inlet flow scaled by position
        constraint flowConstraint {
            (isOpen implies outlet.flowRateMl == inlet.flowRateMl) and
            (not isOpen implies outlet.flowRateMl == 0)
        }
    }

    // Controller that monitors and commands both feeder paths
    part def Controller {
        // References to controlled components
        ref pump1 : Pump;
        ref pump2 : Pump;
        ref valve1 : Valve;
        ref valve2 : Valve;
        ref tank : Tank;
        ref volumeSensor1 : VolumeSensor;
        ref volumeSensor2 : VolumeSensor;

        // Targets.
        attribute tank1TransferMl : Integer;
        attribute tank2TransferMl : Integer;

        // Originals.
        attribute tank1OriginalLevelMl : Integer;
        attribute tank2OriginalLevelMl : Integer;

        // Last observed sensor readings (updated each scan cycle).
        attribute observedLevel1 : Integer = 0;
        attribute observedLevel2 : Integer = 0;

        // Scan frequency.
        attribute scanCycleFrequencyHz : Integer = 10;
        attribute lastScanTimeSeconds : Real = 0;
        attribute currentTimeSeconds : Real = 0;

        // Ports.
        port pump1Port : ModbusMasterPort;
        port pump2Port : ModbusMasterPort;
        port valve1Port : ModbusMasterPort;
        port valve2Port : ModbusMasterPort;
        port volumeSensor1Port : ModbusMasterPort;
        port volumeSensor2Port : ModbusMasterPort;

        action turnOnPump1 {
            item valve1CoilReq : WriteCoilReqMsg;
            assign valve1CoilReq.coilAddress := 0;
            assign valve1CoilReq.coilValue := true;
            send valve1CoilReq via valve1Port.req;
            accept ModbusResMsg via valve1Port.resp;

            item pump1CoilReq : WriteCoilReqMsg;
            assign pump1CoilReq.coilAddress := 0;
            assign pump1CoilReq.coilValue := true;
            send pump1CoilReq via pump1Port.req;
            accept ModbusResMsg via pump1Port.resp;
        }
    
        action turnOffPump1 {
            item pump1CoilReq : WriteCoilReqMsg;
            assign pump1CoilReq.coilAddress := 0;
            assign pump1CoilReq.coilValue := false;
            send pump1CoilReq via pump1Port.req;
            accept ModbusResMsg via pump1Port.resp;

            item valve1CoilReq : WriteCoilReqMsg;
            assign valve1CoilReq.coilAddress := 0;
            assign valve1CoilReq.coilValue := false;
            send valve1CoilReq via valve1Port.req;
            accept ModbusResMsg via valve1Port.resp;
        }

        action turnOnPump2 {
            item valve2CoilReq : WriteCoilReqMsg;
            assign valve2CoilReq.coilAddress := 0;
            assign valve2CoilReq.coilValue := true;
            send valve2CoilReq via valve2Port.req;
            accept ModbusResMsg via valve2Port.resp;

            item pump2CoilReq : WriteCoilReqMsg;
            assign pump2CoilReq.coilAddress := 0;
            assign pump2CoilReq.coilValue := true;
            send pump2CoilReq via pump2Port.req;
            accept ModbusResMsg via pump2Port.resp;
        }

        action turnOffPump2 {
            item pump2CoilReq : WriteCoilReqMsg;
            assign pump2CoilReq.coilAddress := 0;
            assign pump2CoilReq.coilValue := false;
            send pump2CoilReq via pump2Port.req;
            accept ModbusResMsg via pump2Port.resp;

            item valve2CoilReq : WriteCoilReqMsg;
            assign valve2CoilReq.coilAddress := 0;
            assign valve2CoilReq.coilValue := false;
            send valve2CoilReq via valve2Port.req;
            accept ModbusResMsg via valve2Port.resp;
        }

        action openValve1 {
            item valve1CoilReq : WriteCoilReqMsg;
            assign valve1CoilReq.coilAddress := 0;
            assign valve1CoilReq.coilValue := true;
            send valve1CoilReq via valve1Port.req;
            accept ModbusResMsg via valve1Port.resp;
        }

        action closeValve1 {
            item valve1CoilReq : WriteCoilReqMsg;
            assign valve1CoilReq.coilAddress := 0;
            assign valve1CoilReq.coilValue := false;
            send valve1CoilReq via valve1Port.req;
            accept ModbusResMsg via valve1Port.resp;
        }

        action openValve2 {
            item valve2CoilReq : WriteCoilReqMsg;
            assign valve2CoilReq.coilAddress := 0;
            assign valve2CoilReq.coilValue := true;
            send valve2CoilReq via valve2Port.req;
            accept ModbusResMsg via valve2Port.resp;
        }

        action closeValve2 {
            item valve2CoilReq : WriteCoilReqMsg;
            assign valve2CoilReq.coilAddress := 0;
            assign valve2CoilReq.coilValue := false;
            send valve2CoilReq via valve2Port.req;
            accept ModbusResMsg via valve2Port.resp;
        }

        #Neural action def Policy {
            in tank1VolumeMl : Integer;
            in tank2VolumeMl : Integer;
            in tank1TargetTransferMl : Integer;
            in tank2TargetTransferMl : Integer;
            in tank1OriginalMl : Integer;
            in tank2OriginalMl : Integer;
            in done : Boolean;

            out shouldOpenValve1 : Boolean;
            out shouldOpenValve2 : Boolean;
            out shouldTurnOnPump1 : Boolean;
            out shouldTurnOnPump2 : Boolean;
        }

        #NeuralRequirement requirement def 'Neural Controller Soundness' {
            subject p : Policy;
            require constraint {
                (p.tank1OriginalMl - p.tank1VolumeMl < p.tank1TargetTransferMl == p.shouldTurnOnPump1) and 
                (p.shouldTurnOnPump1 == p.shouldOpenValve1) and
                (p.tank2OriginalMl - p.tank2VolumeMl < p.tank2TargetTransferMl == p.shouldTurnOnPump2) and
                (p.shouldTurnOnPump2 == p.shouldOpenValve2)
            }
        }

        action ScanCycle {
            item volume1ReadingReq : ReadInputRegisterReqMsg;
            assign volume1ReadingReq.registerAddress := 0;
            send volume1ReadingReq via volumeSensor1Port.req;
            item volume1Res : ReadInputRegisterRespMsg;
            accept volume1Res : ReadInputRegisterRespMsg via volumeSensor1Port.resp;
            assign observedLevel1 := volume1Res.response;

            item volume2ReadingReq : ReadInputRegisterReqMsg;
            assign volume2ReadingReq.registerAddress := 0;
            send volume2ReadingReq via volumeSensor2Port.req;
            item volume2Res : ReadInputRegisterRespMsg;
            accept volume2Res : ReadInputRegisterRespMsg via volumeSensor2Port.resp;
            assign observedLevel2 := volume2Res.response;

            action policyCall : Policy {
                in tank1VolumeMl := volume1Res.response;
                in tank2VolumeMl := volume2Res.response;
                in tank1TargetTransferMl := tank1TransferMl;
                in tank2TargetTransferMl := tank2TransferMl;
                in tank1OriginalMl := tank1OriginalLevelMl;
                in tank2OriginalMl := tank2OriginalLevelMl;
                in done := (tank1OriginalLevelMl - volume1Res.response >= tank1TargetTransferMl) and
                            (tank2OriginalLevelMl - volume2Res.response >= tank2TargetTransferMl);
            }

            if policyCall.shouldOpenValve1 {
                perform action openValve1;
            }

            if policyCall.shouldTurnOnPump1 {
                perform action turnOnPump1;
            }

            if not policyCall.shouldTurnOnPump1 {
                perform action turnOffPump1;
            }

            if not policyCall.shouldOpenValve1 {
                perform action closeValve1;
            }

            if policyCall.shouldOpenValve2 {
                perform action openValve2;
            }

            if policyCall.shouldTurnOnPump2 {
                perform action turnOnPump2;
            }

            if not policyCall.shouldTurnOnPump2 {
                perform action turnOffPump2;
            }

            if not policyCall.shouldOpenValve2 {
                perform action closeValve2;
            }
        }

        action step {
            in dt : Real;

            assign currentTimeSeconds := currentTimeSeconds + dt;
            if currentTimeSeconds - lastScanTimeSeconds >= 1.0 / scanCycleFrequencyHz {
                assign lastScanTimeSeconds := currentTimeSeconds;
                perform action ScanCycle;
            }
        }

    }

    // The complete system
    part def FillingSystem {
        // Components - feeder path 1
        part pump1 : Pump {
            attribute :>> maxFlowRateMl = 10;  // units per second
        }

        part valve1 : Valve;

        part feederTank1 : Tank {
            attribute :>> capacityMl = 150;
            attribute :>> currentLevelMl = 150;  // starts full
        }

        // Components - feeder path 2
        part pump2 : Pump {
            attribute :>> maxFlowRateMl = 10;  // units per second
        }

        part valve2 : Valve;

        part feederTank2 : Tank {
            attribute :>> capacityMl = 150;
            attribute :>> currentLevelMl = 150;  // starts full
        }

        part fillingTank : Tank {
            attribute :>> capacityMl = 300;
            attribute :>> currentLevelMl = 0;  // starts empty
        }

        part volumeSensor1 : VolumeSensor;
        part volumeSensor2 : VolumeSensor;

        part controller : Controller {
            ref :>> pump1 = system::pump1;
            ref :>> pump2 = system::pump2;
            ref :>> valve1 = system::valve1;
            ref :>> valve2 = system::valve2;
            ref :>> tank = system::fillingTank;
            ref :>> volumeSensor1 = system::volumeSensor1;
            ref :>> volumeSensor2 = system::volumeSensor2;
            #ScenarioInput attribute :>> tank1TransferMl = 50;
            #ScenarioInput attribute :>> tank2TransferMl = 100;
            #ScenarioInput attribute :>> tank1OriginalLevelMl = 150;
            #ScenarioInput attribute :>> tank2OriginalLevelMl = 150;
        }

        #ScenarioConstraint constraint scenarioConstraints {
            controller.tank1TransferMl >= 0 and controller.tank2TransferMl >= 0 and
            controller.tank1OriginalLevelMl >= 0 and controller.tank2OriginalLevelMl >= 0 and
            controller.tank1OriginalLevelMl >= controller.tank1TransferMl and 
            controller.tank2OriginalLevelMl >= controller.tank2TransferMl
        }

        // Flows - feeder path 1
        flow : Fluid from feederTank1.outlet to pump1.inlet;
        flow : Fluid from pump1.outlet to valve1.inlet;
        flow : Fluid from valve1.outlet to fillingTank.inlet;

        // Flows - feeder path 2
        flow : Fluid from feederTank2.outlet to pump2.inlet;
        flow : Fluid from pump2.outlet to valve2.inlet;
        flow : Fluid from valve2.outlet to fillingTank.inlet;

        flow from feederTank1.volumeSensorPort to volumeSensor1.tankConnection;
        flow from feederTank2.volumeSensorPort to volumeSensor2.tankConnection;

        // Modbus topology.
        connect controller.pump1Port to pump1.modbusPort;
        connect controller.pump2Port to pump2.modbusPort;
        connect controller.valve1Port to valve1.modbusPort;
        connect controller.valve2Port to valve2.modbusPort;
        connect controller.volumeSensor1Port to volumeSensor1.modbusPort;
        connect controller.volumeSensor2Port to volumeSensor2.modbusPort;

        // System-level dynamics: tank levels change based on flow
        constraint feeder1DrainRate {
            feederTank1.outlet.flowRateMl == valve1.outlet.flowRateMl
        }

        constraint feeder2DrainRate {
            feederTank2.outlet.flowRateMl == valve2.outlet.flowRateMl
        }

        constraint fillingFillRate {
            fillingTank.inlet.flowRateMl == valve1.outlet.flowRateMl + valve2.outlet.flowRateMl
        }

        #Prohibition
        requirement def 'No Dry Running' {
            doc /* When a sensor reports an empty tank, the controller shall not run its corresponding pump. */
            subject s : FillingSystem;

            require constraint {
                (s.controller.observedLevel1 <= 0 implies not s.pump1.isRunning) and
                (s.controller.observedLevel2 <= 0 implies not s.pump2.isRunning)
            }
        }

        #Prohibition
        requirement def 'No Dead Heading' {
            doc /* When a valve is closed, the controller shall not turn on its corresponding pump. */
            subject s : FillingSystem;

            require constraint {
                (not s.valve1.isOpen implies not s.pump1.isRunning) and
                (not s.valve2.isOpen implies not s.pump2.isRunning)
            }
        }

        #Obligation
        requirement def 'Fluid Transfer Termination Safety' {
            doc /* When the controller observes that the target fluid has been transferred, it shall turn off the pump. */
            subject s : FillingSystem;

            require constraint {
                (s.controller.lastScanTimeSeconds > 0) implies
                ((s.controller.tank1OriginalLevelMl - s.controller.observedLevel1 >= s.controller.tank1TransferMl)
                  implies not s.pump1.isRunning) and
                ((s.controller.tank2OriginalLevelMl - s.controller.observedLevel2 >= s.controller.tank2TransferMl)
                  implies not s.pump2.isRunning)
            }
        }

        #Obligation
        requirement def 'Fluid Transfer Liveness' {
            doc /* When the fluid transferred is below the target amount, the controller shall turn on the pump and open the valve during its next scan cycle. */
            subject s : FillingSystem;

            require constraint {
                (s.controller.lastScanTimeSeconds > 0) implies
                ((s.controller.tank1OriginalLevelMl - s.controller.observedLevel1 < s.controller.tank1TransferMl)
                    implies (s.valve1.isOpen and s.pump1.isRunning) and
                  (s.controller.tank2OriginalLevelMl - s.controller.observedLevel2 < s.controller.tank2TransferMl)
                    implies (s.valve2.isOpen and s.pump2.isRunning))
            }
        }
    }

    part system : FillingSystem;
}
