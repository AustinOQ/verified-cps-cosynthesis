package AirPurifier {
    private import ScalarValues::*;

    metadata def Neural;
    metadata def NeuralRequirement;

    metadata def Prohibition;
    metadata def Obligation;
    metadata def Termination;

    metadata def ScenarioInput;
    metadata def ScenarioConstraint;

    item def AirQualityReading {
        attribute airQualityIndex : Real;
    }

    port def AirQualitySensorPort {
        in item reading : AirQualityReading;
    }

    item def PurificationSignal {
        attribute cleaningRate : Real;
    }

    port def PurificationPort {
        in item purification : PurificationSignal;
    }

    part def Environment {
        attribute airQualityIndex : Real;
        attribute roomVolumeMetersCubed : Real;
        attribute outsidePollutionIndex : Real;
        attribute pollutionLeakRate : Real;
        attribute internalPollutionRate : Real;

        port airQualityPort : ~AirQualitySensorPort;
        port purifierPort : PurificationPort;

        action step {
    in dt : Real;

    if airQualityIndex +
        ((pollutionLeakRate * (outsidePollutionIndex - airQualityIndex)
          + internalPollutionRate
          - purifierPort.purification.cleaningRate)
         * dt) / roomVolumeMetersCubed < 0.0
    {
        assign airQualityIndex := 0.0;
    }
    else {
        assign airQualityIndex :=
            airQualityIndex +
            ((pollutionLeakRate * (outsidePollutionIndex - airQualityIndex)
              + internalPollutionRate
              - purifierPort.purification.cleaningRate)
             * dt) / roomVolumeMetersCubed;
    }
}


        bind airQualityPort.reading.airQualityIndex = airQualityIndex;
    }

    part def AirQualitySensor {
        attribute lastReadingIndex : Real;

        port environmentPort : AirQualitySensorPort;
        port upstreamPort : ~AirQualitySensorPort;

        action step {
            in dt : Real;
            item airReading : AirQualityReading;
            assign airReading.airQualityIndex := environmentPort.reading.airQualityIndex;
            assign lastReadingIndex := airReading.airQualityIndex;
            send airReading via upstreamPort;
        }
    }

    item def PurifierCmd;
    item def PurifierOnCmd :> PurifierCmd;
    item def PurifierResetCmd :> PurifierCmd;

    port def PurifierCmdPort {
        in PurifierCmd;
    }

    part def Purifier {
        attribute cleaningOutput : Real;

        port cmdIn : PurifierCmdPort;
        port flowOut : ~PurificationPort;

        state def PurifierBehavior {
            entry; then reset;
            state reset;
            state on;

            transition reset_to_on
                first reset
                accept PurifierOnCmd via cmdIn
                do action { assign flowOut.purification.cleaningRate := cleaningOutput; }
                then on;

            transition on_to_reset
                first on
                accept PurifierResetCmd via cmdIn
                do action { assign flowOut.purification.cleaningRate := 0.0; }
                then reset;
        }

        exhibit state purifierBehavior : PurifierBehavior;
    }

    part def Controller {
        attribute targetAirQualityIndex : Real;
        attribute toleranceIndex : Real;
        attribute purifierOn : Boolean;

        port purifierControlPort : ~PurifierCmdPort;
        port sensorPort : AirQualitySensorPort;

        #Neural action def Policy {
            in targetAirQualityIndex : Real;
            in airQualityIndex : Real;
            in withinTarget : Boolean;

            out purifierState : Boolean;
        }

        #NeuralRequirement requirement def 'Neural Controller Soundness' {
            subject p : Policy;
            require constraint {
                (p.targetAirQualityIndex < p.airQualityIndex - toleranceIndex == p.purifierState) and
                (not (p.withinTarget and p.purifierState))
            }
        }

        action step {
            in dt : Real;

            item reading : AirQualityReading;
            accept reading : AirQualityReading via sensorPort.reading;

            action policyCall : Policy {
                in targetAirQualityIndex := targetAirQualityIndex;
                in airQualityIndex := reading.airQualityIndex;
                in withinTarget := targetAirQualityIndex >= reading.airQualityIndex - toleranceIndex and not purifierOn;
            }

            assign purifierOn := policyCall.purifierState;

            if not purifierOn {
                item resetCmd : PurifierResetCmd;
                send resetCmd via purifierControlPort;
            }

            if policyCall.purifierState {
                item onCmd : PurifierOnCmd;
                send onCmd via purifierControlPort;
            }
        }
    }

    part def AirPurifierSystem {
        part environment : Environment;
        part sensor : AirQualitySensor;
        part purifier : Purifier;
        part controller : Controller;

        flow from purifier.flowOut to environment.purifierPort;
        flow from environment.airQualityPort to sensor.environmentPort;

        connect controller.purifierControlPort to purifier.cmdIn;
        connect sensor.upstreamPort to controller.sensorPort;

        attribute currentTime : Real = 0;
        attribute lastObservedAirQuality : Real = sensor.lastReadingIndex;

        action step {
            in dt : Real;
            assign currentTime := currentTime + dt;
        }

        #Prohibition
requirement def 'Do Not Purify When Air Is Good' {
    subject s : AirPurifierSystem;

    require constraint {
        currentTime > 0 implies
            (s.lastObservedAirQuality < s.controller.targetAirQualityIndex - s.controller.toleranceIndex
                implies not s.controller.purifierOn)
    }
}

#Obligation
requirement def 'Purify When Air Is Bad' {
    subject s : AirPurifierSystem;

    require constraint {
        currentTime > 0 implies
            (s.lastObservedAirQuality > s.controller.targetAirQualityIndex + s.controller.toleranceIndex
                implies s.controller.purifierOn)
    }
}


        #Obligation
        requirement def 'Air Quality Stays Nonnegative' {
            subject s : AirPurifierSystem;

            require constraint {
                s.environment.airQualityIndex >= 0
            }
        }
    }

    part system : AirPurifierSystem {
        part :>> environment {
            attribute :>> airQualityIndex = 80.0;
            attribute :>> roomVolumeMetersCubed = 25.0;
            #ScenarioInput attribute :>> outsidePollutionIndex = 120.0;
            attribute :>> pollutionLeakRate = 1.5;
            #ScenarioInput attribute :>> internalPollutionRate = 2.0;
        }

        part :>> purifier {
            attribute :>> cleaningOutput = 12.0;
        }

        part :>> controller {
            #ScenarioInput attribute :>> targetAirQualityIndex = 40.0;
            attribute :>> toleranceIndex = 5.0;
        }

        #ScenarioConstraint constraint scenarioConstraint {
            controller.targetAirQualityIndex >= 10.0 and
            controller.targetAirQualityIndex <= 100.0 and
            environment.outsidePollutionIndex >= 0.0 and
            environment.outsidePollutionIndex <= 300.0 and
            environment.internalPollutionRate >= 0.0 and
            environment.internalPollutionRate <= 20.0
        }
    }
}
