package CruiseControlContinuous {
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
        attribute gapMeters : Real;
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

    // Continuous drive command: a single normalized force in [-100, 100].
    // + = accelerate (engine), - = decelerate (brake).
    item def DriveCommand {
        attribute normalizedForce : Real;
    }

    port def DriveCmdPort {
        in item cmd : DriveCommand;
    }

    part def Vehicle {
        attribute speedMps : Real;
        attribute massKg : Real;
        attribute dragCoefficient : Real;
        attribute roadGradePercent : Real;
        attribute gapMeters : Real;
        attribute leadSpeedMps : Real;

        port speedSensorPort : ~SpeedSensorPort;
        port drivePort : DriveForcePort;

        action step {
            in dt : Real;
            // Net force = drive force - aerodynamic drag - grade resistance.
            // drive force is signed: positive accelerates, negative brakes.
            assign speedMps :=
                speedMps +
                ((drivePort.drive.forceNewtons
                  - dragCoefficient * speedMps * speedMps
                  - massKg * 9.81 * roadGradePercent / 100.0)
                 * dt) / massKg;

            // Gap to lead vehicle: closes when ego is faster, opens when slower.
            assign gapMeters := gapMeters + (leadSpeedMps - speedMps) * dt;
        }

        bind speedSensorPort.reading.speedMps = speedMps;
        bind speedSensorPort.reading.gapMeters = gapMeters;
    }

    part def SpeedSensor {
        attribute lastReadingMps : Real;
        attribute lastGapMeters : Real;

        port vehiclePort : SpeedSensorPort;
        port upstreamPort : ~SpeedSensorPort;

        action step {
            in dt : Real;
            item speedReading : SpeedReading;
            assign speedReading.speedMps := vehiclePort.reading.speedMps;
            assign speedReading.gapMeters := vehiclePort.reading.gapMeters;
            assign lastReadingMps := speedReading.speedMps;
            assign lastGapMeters := speedReading.gapMeters;
            send speedReading via upstreamPort;
        }
    }

    // Continuous actuator: scales the normalized command [-100, 100] to a
    // physical force, asymmetric (engine vs brake limits). No on/off state.
    part def DriveActuator {
        attribute maxEngineForceNewtons : Real;   // positive
        attribute maxBrakeForceNewtons : Real;    // positive magnitude
        attribute appliedForceNewtons : Real =
            (cmdIn.cmd.normalizedForce >= 0
                ? cmdIn.cmd.normalizedForce * maxEngineForceNewtons / 100.0
                : cmdIn.cmd.normalizedForce * maxBrakeForceNewtons / 100.0);

        port cmdIn : DriveCmdPort;
        port forceOut : ~DriveForcePort;

        bind forceOut.drive.forceNewtons = appliedForceNewtons;
    }

    part def Controller {
        attribute targetSpeedMps : Real;
        attribute toleranceMps : Real;
        attribute safeFollowingDistanceMeters : Real;
        attribute onThreshold : Real;
        attribute propagationDelay : Real = 3;
        attribute commandedForce : Real;

        // throttle/brake "on" derived from the continuous command via onThreshold.
        attribute throttleOn : Boolean = commandedForce > onThreshold;
        attribute brakeOn : Boolean = commandedForce < (0 - onThreshold);

        port commandOut : ~DriveCmdPort;
        port speedometerPort : SpeedSensorPort;

        // Continuous command output: the normalized force is published on the
        // output port (a bind, not a discrete send) so it propagates by flow.
        bind commandOut.cmd.normalizedForce = commandedForce;

        #Neural action def Policy {
            in currentSpeedMps : Real;
            in targetSpeed : Real;
            in gapMeters : Real;
            #Completion in done : Boolean;

            out driveForce : Real;
        }

        #NeuralRequirement requirement def 'Neural Controller Soundness' {
            subject p : Policy;
            require constraint {
                (p.targetSpeed > p.currentSpeedMps + toleranceMps and p.gapMeters >= safeFollowingDistanceMeters == p.driveForce > onThreshold) and
                (p.targetSpeed < p.currentSpeedMps - toleranceMps or p.gapMeters < safeFollowingDistanceMeters == p.driveForce < (0 - onThreshold)) and
                (p.driveForce >= (0 - 100) and p.driveForce <= 100)
            }
        }

        action step {
            in dt : Real;

            item reading : SpeedReading;
            accept reading : SpeedReading via speedometerPort.reading;

            action policyCall : Policy {
                in currentSpeedMps := reading.speedMps;
                in targetSpeed := targetSpeedMps;
                in gapMeters := reading.gapMeters;
                in done := targetSpeedMps <= reading.speedMps + toleranceMps and
                    targetSpeedMps >= reading.speedMps - toleranceMps and not throttleOn and not brakeOn;
            }

            assign commandedForce := policyCall.driveForce;
        }
    }

    part def CruiseControlSystem {
        part vehicle : Vehicle;
        part speedSensor : SpeedSensor;
        part drive : DriveActuator;
        part controller : Controller;

        flow from drive.forceOut to vehicle.drivePort;
        flow from vehicle.speedSensorPort to speedSensor.vehiclePort;
        flow from controller.commandOut to drive.cmdIn;

        connect speedSensor.upstreamPort to controller.speedometerPort;

        attribute currentTime : Real = 0;
        attribute lastObservedSpeed : Real = speedSensor.lastReadingMps;
        attribute lastObservedGap : Real = speedSensor.lastGapMeters;

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

        #Prohibition
        requirement def 'No Throttle When Too Close' {
            doc /* The controller shall not apply throttle when the gap to the lead vehicle is below the safe following distance. */
            subject s : CruiseControlSystem;

            require constraint {
                s.lastObservedGap < s.controller.safeFollowingDistanceMeters implies not s.controller.throttleOn
            }
        }

        #Prohibition
        requirement def 'No Throttle When Overspeeding' {
            doc /* The controller shall not apply throttle when speed exceeds target plus tolerance. */
            subject s : CruiseControlSystem;

            require constraint {
                currentTime > 0 implies
                    (s.lastObservedSpeed > s.controller.targetSpeedMps + s.controller.toleranceMps
                        implies not s.controller.throttleOn)
            }
        }

        #Obligation
        requirement def 'Accelerate When Below Target' {
            doc /* When speed is below target minus tolerance and the gap is safe, throttle should be on. */
            subject s : CruiseControlSystem;

            require constraint {
                currentTime > 0 implies
                    (s.lastObservedSpeed < s.controller.targetSpeedMps - s.controller.toleranceMps
                        and s.lastObservedGap >= s.controller.safeFollowingDistanceMeters
                        implies s.controller.throttleOn)
            }
        }

        #Obligation
        requirement def 'Brake When Above Target Or Too Close' {
            doc /* When speed is above target plus tolerance or the gap is below safe following distance, brake should be on. */
            subject s : CruiseControlSystem;

            require constraint {
                currentTime > 0 implies
                    (s.lastObservedSpeed > s.controller.targetSpeedMps + s.controller.toleranceMps
                        or s.lastObservedGap < s.controller.safeFollowingDistanceMeters
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
            attribute :>> gapMeters = 100.0;
            #ScenarioInput attribute :>> leadSpeedMps = 20.0;
        }

        part :>> drive {
            attribute :>> maxEngineForceNewtons = 5000.0;
            attribute :>> maxBrakeForceNewtons = 8000.0;
        }

        part :>> controller {
            #ScenarioInput attribute :>> targetSpeedMps = 25.0;
            attribute :>> toleranceMps = 2.0;
            attribute :>> safeFollowingDistanceMeters = 10.0;
            attribute :>> onThreshold = 0.1;
        }

        #ScenarioConstraint constraint scenarioConstraint {
            controller.targetSpeedMps >= 5.0 and
            controller.targetSpeedMps <= 25.0 and
            vehicle.leadSpeedMps >= 5.0 and
            vehicle.leadSpeedMps <= 20.0
        }
    }
}
