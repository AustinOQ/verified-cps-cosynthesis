package CruiseControl {
    private import ScalarValues::*;

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

    item def SpeedReading {
        attribute speedMps : Real;
    }

    port def SpeedSensorPort {
        in item reading : SpeedReading;
    }

    item def DriveForce {
        attribute forceNewtons : Real;
    }

    port def DriveForcePort {
        in item drive : DriveForce;
    }

    item def ActuatorCmd;
    item def ActuatorOnCmd :> ActuatorCmd;
    item def ActuatorResetCmd :> ActuatorCmd;

    port def ActuatorCmdPort {
        in ActuatorCmd;
    }

    part def Vehicle {
        attribute speedMps : Real;
        attribute massKg : Real;
        attribute dragCoefficient : Real;
        attribute roadGradePercent : Real;

        port speedSensorPort : ~SpeedSensorPort;
        port enginePort : DriveForcePort;
        port brakePort : DriveForcePort;

        action step {
            in dt : Real;
            // Net force = engine + brake - aerodynamic drag - grade resistance.
            // Brake force is negative when braking.
            assign speedMps :=
                speedMps +
                ((enginePort.drive.forceNewtons + brakePort.drive.forceNewtons
                  - dragCoefficient * speedMps * speedMps
                  - massKg * 9.81 * roadGradePercent / 100.0)
                 * dt) / massKg;
        }

        bind speedSensorPort.reading.speedMps = speedMps;
    }

    part def SpeedSensor {
        attribute lastReadingMps : Real;

        port vehiclePort : SpeedSensorPort;
        port upstreamPort : ~SpeedSensorPort;

        action step {
            in dt : Real;
            item speedReading : SpeedReading;
            assign speedReading.speedMps := vehiclePort.reading.speedMps;
            assign lastReadingMps := speedReading.speedMps;
            send speedReading via upstreamPort;
        }
    }

    part def Actuator {
        attribute maxForceNewtons : Real;

        port cmdIn : ActuatorCmdPort;
        port forceOut : ~DriveForcePort;

        state def ActuatorBehavior {
            entry; then off;
            state off;
            state on;

            transition off_to_on
                first off
                accept ActuatorOnCmd via cmdIn
                do action { assign forceOut.drive.forceNewtons := maxForceNewtons; }
                then on;

            transition on_to_off
                first on
                accept ActuatorResetCmd via cmdIn
                do action { assign forceOut.drive.forceNewtons := 0.0; }
                then off;
        }

        exhibit state actuatorBehavior : ActuatorBehavior;
    }

    part def Controller {
        attribute targetSpeedMps : Real;
        attribute toleranceMps : Real;
        attribute throttleOn : Boolean;
        attribute brakeOn : Boolean;

        port throttleCmdPort : ~ActuatorCmdPort;
        port brakeCmdPort : ~ActuatorCmdPort;
        port speedometerPort : SpeedSensorPort;

        #Neural action def Policy {
            in currentSpeedMps : Real;
            in targetSpeed : Real;
            in done : Boolean;

            out applyThrottle : Boolean;
            out applyBrake : Boolean;
        }

        #NeuralRequirement requirement def 'Neural Controller Soundness' {
            subject p : Policy;
            require constraint {
                (p.targetSpeed > p.currentSpeedMps + toleranceMps == p.applyThrottle) and
                (p.currentSpeedMps > p.targetSpeed + toleranceMps == p.applyBrake) and
                (not (p.applyThrottle and p.applyBrake))
            }
        }

        action step {
            in dt : Real;

            item reading : SpeedReading;
            accept reading : SpeedReading via speedometerPort.reading;

            action policyCall : Policy {
                in currentSpeedMps := reading.speedMps;
                in targetSpeed := targetSpeedMps;
                in done := targetSpeedMps <= reading.speedMps + toleranceMps and
                    targetSpeedMps >= reading.speedMps - toleranceMps;
            }

            assign throttleOn := policyCall.applyThrottle;
            assign brakeOn := policyCall.applyBrake;

            if not throttleOn {
                item resetCmd : ActuatorResetCmd;
                send resetCmd via throttleCmdPort;
            }

            if not brakeOn {
                item resetCmd : ActuatorResetCmd;
                send resetCmd via brakeCmdPort;
            }

            if policyCall.applyThrottle {
                item onCmd : ActuatorOnCmd;
                send onCmd via throttleCmdPort;
            }

            if policyCall.applyBrake {
                item onCmd : ActuatorOnCmd;
                send onCmd via brakeCmdPort;
            }
        }
    }

    part def CruiseControlSystem {
        part vehicle : Vehicle;
        part speedSensor : SpeedSensor;
        part engine : Actuator;
        part brake : Actuator;
        part controller : Controller;

        flow from engine.forceOut to vehicle.enginePort;
        flow from brake.forceOut to vehicle.brakePort;
        flow from vehicle.speedSensorPort to speedSensor.vehiclePort;

        connect controller.throttleCmdPort to engine.cmdIn;
        connect controller.brakeCmdPort to brake.cmdIn;
        connect speedSensor.upstreamPort to controller.speedometerPort;

        attribute currentTime : Real = 0;
        attribute lastObservedSpeed : Real = speedSensor.lastReadingMps;

        action step {
            in dt : Real;
            assign currentTime := currentTime + dt;
        }

        #Prohibition
        requirement def 'No Simultaneous Throttle and Brake' {
            doc /* The controller shall not apply throttle and brake at the same time. */
            subject s : CruiseControlSystem;

            require constraint {
                not (s.controller.throttleOn and s.controller.brakeOn)
            }
        }

        #Obligation
        requirement def 'No Throttle Above Target' {
            doc /* The controller shall not apply throttle when speed is above the target speed. */
            subject s : CruiseControlSystem;

            require constraint {
                currentTime > 0 implies
                    (s.lastObservedSpeed > s.controller.targetSpeedMps
                        implies not s.controller.throttleOn)
            }
        }

        #Obligation
        requirement def 'No Brake Below Target' {
            doc /* The controller shall not brake when speed is below the target speed. */
            subject s : CruiseControlSystem;

            require constraint {
                currentTime > 0 implies
                    (s.lastObservedSpeed < s.controller.targetSpeedMps
                        implies not s.controller.brakeOn)
            }
        }

        #Obligation
        requirement def 'Accelerate When Below Target' {
            doc /* When speed is below target minus tolerance, throttle should be on. */
            subject s : CruiseControlSystem;

            require constraint {
                currentTime > 0 implies
                    (s.lastObservedSpeed < s.controller.targetSpeedMps - s.controller.toleranceMps
                        implies s.controller.throttleOn)
            }
        }

        #Obligation
        requirement def 'Brake When Above Target' {
            doc /* When speed is above target plus tolerance, brake should be on. */
            subject s : CruiseControlSystem;

            require constraint {
                currentTime > 0 implies
                    (s.lastObservedSpeed > s.controller.targetSpeedMps + s.controller.toleranceMps
                        implies s.controller.brakeOn)
            }
        }
    }

    part system : CruiseControlSystem {
        part :>> vehicle {
            attribute :>> speedMps = 0.0;
            attribute :>> massKg = 1500.0;
            attribute :>> dragCoefficient = 0.4;
            attribute :>> roadGradePercent = 0.0;
        }

        part :>> engine {
            attribute :>> maxForceNewtons = 5000.0;
        }

        part :>> brake {
            attribute :>> maxForceNewtons = -8000.0;
        }

        part :>> controller {
            #ScenarioInput attribute :>> targetSpeedMps = 25.0;
            attribute :>> toleranceMps = 1.0;
        }

        #ScenarioConstraint constraint scenarioConstraint {
            controller.targetSpeedMps >= 5.0 and
            controller.targetSpeedMps <= 25.0
        }
    }
}
