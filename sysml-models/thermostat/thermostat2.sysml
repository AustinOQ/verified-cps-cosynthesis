package Thermostat {
    private import ScalarValues::*;

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

        port temperaturePort : ~TemperatureSensorPort;
        port heaterPort : HeatingCoolingPort;
        port acPort : HeatingCoolingPort;

        action step {
            in dt : Real;

            // Update temperature.
            // Delta T = Energy Added / (mass * specific heat capacity).
            // TODO: Model heat loss.
            assign temperatureCelcius := 
                temperatureCelcius + 
                ((heaterPort.heat.rateWatts + acPort.heat.rateWatts) * dt) / (volumeMetersCubed * 1.2 * 1005);
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
        
        port heaterControlPort : ~HeatingCoolingCmdPort;
        port acControlPort : ~HeatingCoolingCmdPort;
        port thermometerPort : TemperatureSensorPort;

        state def ControllerBehavior {
            entry; then waiting;

            state waiting;
            state heating;
            state cooling;

            transition waiting_to_heating
                first waiting
                accept temperatureReading : TemperatureReading via thermometerPort 
                if temperatureReading.temperatureCelcius < setPointCelcius - toleranceCelcius
                do action {
                    item onCmd : HeatingCoolingOnCmd;
                    send onCmd via heaterControlPort; 
                }
                then heating;
            
            transition heating_to_waiting
                first heating
                accept temperatureReading : TemperatureReading via thermometerPort
                if temperatureReading.temperatureCelcius >= setPointCelcius - toleranceCelcius
                do action {
                    item resetCmd : HeatingCoolingResetCmd;
                    send resetCmd via heaterControlPort;
                }
                then waiting;
            
            transition waiting_to_cooling
                first waiting
                accept temperatureReading : TemperatureReading via thermometerPort
                if temperatureReading.temperatureCelcius > setPointCelcius + toleranceCelcius
                do action {
                    item onCmd : HeatingCoolingOnCmd;
                    send onCmd via acControlPort;
                }
                then cooling;
            
            transition cooling_to_waiting
                first cooling
                accept temperatureReading : TemperatureReading via thermometerPort 
                if temperatureReading.temperatureCelcius <= setPointCelcius + toleranceCelcius
                do action {
                    item resetCmd : HeatingCoolingResetCmd;
                    send resetCmd via acControlPort;
                }
                then waiting;
        }

        exhibit state controllerBehavior : ControllerBehavior;
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
    }

    part system : HeatingCoolingSystem {
        part :>> environment {
            attribute :>> temperatureCelcius = 23.9;
            attribute :>> volumeMetersCubed = 20;
        }

        part :>> ac {
            attribute :>> heatOutputWatts = -2000;
        }

        part :>> heater {
            attribute :>> heatOutputWatts = 2000;
        }

        part :>> controller {
            attribute :>> setPointCelcius = 18.3;
            attribute :>> toleranceCelcius = 1;
        }
    }
}