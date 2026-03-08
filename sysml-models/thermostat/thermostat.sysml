package Thermostat {
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

    item def TemperatureReading {
        attribute temperatureCelcius : Real;
    }

    port def TemperatureSensorPort {
        in item reading : TemperatureReading;
    }

    item def ThermalEnergy {
        attribute rateWatts : Real;
    }

    port def HeatingCoolingPort {
        in item heat : ThermalEnergy;
    }

    part def Environment {
        attribute temperatureCelcius : Real;
        attribute volumeMetersCubed : Real;
        attribute outsideTemperatureCelcius : Real;
        attribute heatLossCoefficientWattsPerCelcius : Real;

        port temperaturePort : ~TemperatureSensorPort;
        port heaterPort : HeatingCoolingPort;
        port acPort : HeatingCoolingPort;

        action step {
            in dt : Real;

            // Update temperature.
            // Delta T = (Energy Added - Heat Loss) / (mass * specific heat capacity).
            // Heat loss via Newton's law of cooling: Q_loss = U*A * (T_inside - T_outside).
            assign temperatureCelcius :=
                temperatureCelcius +
                ((heaterPort.heat.rateWatts + acPort.heat.rateWatts
                  - heatLossCoefficientWattsPerCelcius * (temperatureCelcius - outsideTemperatureCelcius))
                 * dt) / (volumeMetersCubed * 1.2 * 1005);
        }


        // Send updated temperature reading.
        bind temperaturePort.reading.temperatureCelcius = temperatureCelcius;
    }

    part def TemperatureSensor {
        port environmentPort : TemperatureSensorPort;
        port upstreamPort : ~TemperatureSensorPort;

        action step {
            in dt : Real;
            item temperatureReading : TemperatureReading;
            assign temperatureReading.temperatureCelcius := environmentPort.reading.temperatureCelcius;
            send temperatureReading via upstreamPort;
        }
    }

    item def HeatingCoolingCmd;
    item def HeatingCoolingOnCmd :> HeatingCoolingCmd;
    item def HeatingCoolingResetCmd :> HeatingCoolingCmd;

    port def HeatingCoolingCmdPort {
        in HeatingCoolingCmd;
    }

    part def HeatingCoolingSource {
        attribute heatOutputWatts : Real;

        port cmdIn : HeatingCoolingCmdPort;
        port heatOut : ~HeatingCoolingPort;

        state def HeaterBehavior {
            entry; then reset;
            state reset;
            state on;

            transition reset_to_on
                first reset
                accept HeatingCoolingOnCmd via cmdIn
                do action { assign heatOut.heat.rateWatts := heatOutputWatts; }
                then on;

            transition on_to_reset
                first on
                accept HeatingCoolingResetCmd via cmdIn
                do action { assign heatOut.heat.rateWatts := 0.0; }
                then reset;
        }

        exhibit state heaterBehavior : HeaterBehavior;
    }

    part def Controller {
        attribute setPointCelcius : Real;
        attribute toleranceCelcius : Real;
        attribute heaterOn : Boolean;
        attribute acOn : Boolean;
        
        port heaterControlPort : ~HeatingCoolingCmdPort;
        port acControlPort : ~HeatingCoolingCmdPort;
        port thermometerPort : TemperatureSensorPort;

        #Neural action def Policy {
            in setPoint : Real;
            in temperatureCelcius : Real;
            in done : Boolean;

            out heaterState : Boolean;
            out acState : Boolean;
        }

        #NeuralRequirement requirement def 'Neural Controller Soundness' {
            subject p : Policy;
            require constraint {
                (p.setPoint > p.temperatureCelcius + toleranceCelcius == p.heaterState) and 
                (p.setPoint < p.temperatureCelcius - toleranceCelcius == p.acState) and 
                (not (p.heaterState and p.acState))
            }
        }

        action step {
            in dt : Real;

            item reading : TemperatureReading;
            accept reading : TemperatureReading via thermometerPort.reading;

            action policyCall : Policy {
                in setPoint := setPointCelcius;
                in temperatureCelcius := reading.temperatureCelcius;
                in done := setPointCelcius <= reading.temperatureCelcius + toleranceCelcius and
                setPointCelcius >= reading.temperatureCelcius - toleranceCelcius and not heaterOn and not acOn;
            }

            // Need to turn off before we turn on. 

            assign heaterOn := policyCall.heaterState;
            assign acOn := policyCall.acState;
            if not heaterOn {
                item resetCmd : HeatingCoolingResetCmd;
                send resetCmd via heaterControlPort;
            }

            if not acOn {
                item resetCmd : HeatingCoolingResetCmd;
                send resetCmd via acControlPort;
            }

            if policyCall.heaterState {
                item onCmd : HeatingCoolingOnCmd;
                send onCmd via heaterControlPort;
            }
            
            if policyCall.acState {
                item onCmd : HeatingCoolingOnCmd;
                send onCmd via acControlPort;
            }
        }
    }

    part def HeatingCoolingSystem {
        part environment : Environment;
        part thermometer : TemperatureSensor;
        part ac : HeatingCoolingSource;
        part heater : HeatingCoolingSource;
        part controller : Controller;

        flow from ac.heatOut to environment.acPort;
        flow from heater.heatOut to environment.heaterPort;
        flow from environment.temperaturePort to thermometer.environmentPort;

        connect controller.acControlPort to ac.cmdIn;
        connect controller.heaterControlPort to heater.cmdIn;
        connect thermometer.upstreamPort to controller.thermometerPort;

        attribute currentTime : Real = 0;

        action step {
            in dt : Real;
            assign currentTime := currentTime + dt;
        }

        #Prohibition
        requirement def 'No Simultaneous Heating and Cooling' {
            doc /* The controller shall not turn on both the heater and AC at the same time. */
            subject s : HeatingCoolingSystem;

            require constraint {
                not (s.controller.heaterOn and s.controller.acOn)
            }
        }

        #Obligation
        requirement def 'Heat When Cold' {
            doc /* When the temperature is below the setpoint minus tolerance, the heater shall be on. */
            subject s : HeatingCoolingSystem;

            require constraint {
                currentTime > 0 implies 
                    (s.environment.temperatureCelcius < s.controller.setPointCelcius - s.controller.toleranceCelcius
                        implies s.controller.heaterOn)
            }
        }

        #Obligation
        requirement def 'Cool When Hot' {
            doc /* When the temperature is above the setpoint plus tolerance, the AC shall be on. */
            subject s : HeatingCoolingSystem;

            require constraint {
                currentTime > 0 implies
                    (s.environment.temperatureCelcius > s.controller.setPointCelcius + s.controller.toleranceCelcius
                        implies s.controller.acOn)
            }
        }
    }

    part system : HeatingCoolingSystem {
        part :>> environment {
            attribute :>> temperatureCelcius = 23.9;
            attribute :>> volumeMetersCubed = 20;
            #ScenarioInput attribute :>> outsideTemperatureCelcius = 5.0;
            attribute :>> heatLossCoefficientWattsPerCelcius = 10.0;
        }

        part :>> ac {
            attribute :>> heatOutputWatts = -2000;
        }

        part :>> heater {
            attribute :>> heatOutputWatts = 2000;
        }

        part :>> controller {
            #ScenarioInput attribute :>> setPointCelcius = 18.3;
            attribute :>> toleranceCelcius = 1;
        }

        #ScenarioConstraint constraint scenarioConstraint {
            controller.setPointCelcius >= 12.78 and 
            controller.setPointCelcius <= 37.78 and 
            environment.outsideTemperatureCelcius <= 50 and 
            environment.outsideTemperatureCelcius >= -30
        }
    }
}