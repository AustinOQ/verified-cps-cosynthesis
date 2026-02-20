package DigitalThermostatSystem {

    private import ScalarValues::*;
    private import SI::*;

    item def Heat;

    // Port definition carrying temperature and heat flow
    port def ThermalPort {
        attribute temperature : Real;   // e.g., °C
        attribute heatFlow    : Real;   // positive adds heat to the connected element
    }

    // Physical room model: temperature changes based on net heat flow
    part def Room {
        port thermal : ThermalPort;

        // State variable 
        attribute T_room : Real;

        // Simple physical parameters
        attribute thermalMass : Real;     // bigger = slower temp change
        attribute leakRate    : Real;     // how strongly room moves toward ambient
        attribute T_ambient   : Real;     // outside/ambient temperature

        // Keep port temperature consistent with room temperature
        constraint portTemp {
            thermal.temperature == T_room
        }

        // Dynamics 
        // Room warms when heatFlow > 0, cools toward ambient based on leakRate.
        constraint dynamics {
            der(T_room) == (thermal.heatFlow / thermalMass) - leakRate * (T_room - T_ambient)
        }
    }

    // Heater actuator: ON/OFF produces heatFlow
    part def Heater {
        port thermal : ThermalPort;

        // Cyber command (output of thermostat)
        attribute H : Boolean;

        // Physical capability
        attribute maxHeatFlow : Real;

        // When ON -> provides maxHeatFlow, else 0
        constraint heatConstraint {
            thermal.heatFlow == (H ? maxHeatFlow : 0)
        }
    }

    // Thermostat controller behavior 
    state def ThermostatBehavior {
        state OFF;
        state ON;

        // OFF -> ON: switch is ON and room temp below setpoint
        transition off_to_on
            first OFF
            if (S and (T_room < T_set))
            then ON;

        // ON -> OFF: switch OFF OR reached setpoint
        transition on_to_off
            first ON
            if ((not S) or (T_room >= T_set))
            then OFF;
    }

    // Controller: monitors T_room and commands heater H
    part def ThermostatController {

        // Inputs (user + sensor)
        attribute S      : Boolean;  // switch
        attribute T_set  : Real;     // setpoint
        attribute T_room : Real;     // sensor reading

        // Output (to heater)
        attribute H      : Boolean;

        // Behavior
        exhibit state behavior : ThermostatBehavior;

        // Control logic per state 
        constraint offConstraint {
            behavior.OFF implies (H == false)
        }

        constraint onConstraint {
            behavior.ON implies (H == true)
        }
    }

    // Complete thermostat system: connect controller + heater + room
    part def ThermostatSystem {

        part room : Room {
            attribute :>> T_room       = 18.0;
            attribute :>> T_ambient    = 10.0;
            attribute :>> thermalMass  = 50.0;
            attribute :>> leakRate     = 0.01;
        }

        part heater : Heater {
            attribute :>> maxHeatFlow = 30.0;
        }

        part controller : ThermostatController {
            attribute :>> S     = true;
            attribute :>> T_set = 22.0;

            // bind controller’s sensed temp to room temp
            attribute :>> T_room = system::room.T_room;
        }

        // Connect heater heat into the room (flow connection)
        // This says heat flows from heater to room via thermal ports.
        flow : Heat from heater.thermal to room.thermal;

        // Bind heater command to controller output
        constraint commandBinding {
            heater.H == controller.H
        }

        constraint tempConsistency {
            heater.thermal.temperature == room.thermal.temperature
        }
    }

    part system : ThermostatSystem;
}