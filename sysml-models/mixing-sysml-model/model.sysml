package TankFillingSystem {

    // Standard library imports
    private import ScalarValues::*;
    private import SI::*;

    // Item that flows through the system
    item def Fluid;

    // Port definition with flow rate
    port def FluidPort {
        attribute flowRate : Real;
    }

    // Tank with level tracking
    part def Tank {
        port outlet : FluidPort;
        port inlet : FluidPort;

        attribute capacity : Real;
        attribute currentLevel : Real;
        attribute isFull : Boolean = currentLevel >= capacity;
        attribute isEmpty : Boolean = currentLevel <= 0;

        // Dynamics: level changes based on net flow
        constraint dynamics {
            der(currentLevel) == inlet.flowRate - outlet.flowRate
        }
    }

    // Pump with on/off state and flow rate
    part def Pump {
        port inlet : FluidPort;
        port outlet : FluidPort;

        attribute isRunning : Boolean;
        attribute maxFlowRate : Real;

        // When running, outlet flow equals max rate; otherwise zero
        constraint flowConstraint {
            outlet.flowRate == (isRunning ? maxFlowRate : 0)
        }
    }

    // Valve with position control
    part def Valve {
        port inlet : FluidPort;
        port outlet : FluidPort;

        attribute isOpen : Boolean;
        attribute position : Real;  // 0.0 = closed, 1.0 = fully open

        // Outlet flow is inlet flow scaled by position
        constraint flowConstraint {
            outlet.flowRate == (isOpen ? inlet.flowRate * position : 0)
        }
    }

    // State machine for controller behavior
    state def ControllerBehavior {
        state idle;
        state filling;
        state full;

        transition idle_to_filling
            first idle
            accept startCommand
            then filling;

        transition filling_to_full
            first filling
            if tankIsFull
            then full;

        transition full_to_idle
            first full
            accept resetCommand
            then idle;
    }

    // Controller that monitors and commands both feeder paths
    part def Controller {
        // References to controlled components
        ref pump1 : Pump;
        ref pump2 : Pump;
        ref valve1 : Valve;
        ref valve2 : Valve;
        ref tank : Tank;

        // Commands
        attribute startCommand : Boolean;
        attribute resetCommand : Boolean;

        // Monitored value
        attribute tankIsFull : Boolean = tank.isFull;

        // Behavior
        exhibit state behavior : ControllerBehavior;

        // Control logic per state
        constraint idleConstraint {
            behavior.idle implies (pump1.isRunning == false and valve1.isOpen == false and pump2.isRunning == false and valve2.isOpen == false)
        }

        constraint fillingConstraint {
            behavior.filling implies (pump1.isRunning == true and valve1.isOpen == true and pump2.isRunning == true and valve2.isOpen == true)
        }

        constraint fullConstraint {
            behavior.full implies (pump1.isRunning == false and valve1.isOpen == false and pump2.isRunning == false and valve2.isOpen == false)
        }
    }

    // The complete system
    part def FillingSystem {
        // Components - feeder path 1
        part pump1 : Pump {
            attribute :>> maxFlowRate = 10.0;  // units per second
        }

        part valve1 : Valve {
            attribute :>> position = 1.0;  // fully open when open
        }

        part feederTank1 : Tank {
            attribute :>> capacity = 1000.0;
            attribute :>> currentLevel = 1000.0;  // starts full
        }

        // Components - feeder path 2
        part pump2 : Pump {
            attribute :>> maxFlowRate = 10.0;  // units per second
        }

        part valve2 : Valve {
            attribute :>> position = 1.0;  // fully open when open
        }

        part feederTank2 : Tank {
            attribute :>> capacity = 1000.0;
            attribute :>> currentLevel = 1000.0;  // starts full
        }

        part fillingTank : Tank {
            attribute :>> capacity = 100.0;
            attribute :>> currentLevel = 0.0;  // starts empty
        }

        part controller : Controller {
            ref :>> pump1 = system::pump1;
            ref :>> pump2 = system::pump2;
            ref :>> valve1 = system::valve1;
            ref :>> valve2 = system::valve2;
            ref :>> tank = system::fillingTank;
        }

        // Flows - feeder path 1
        flow : Fluid from feederTank1.outlet to pump1.inlet;
        flow : Fluid from pump1.outlet to valve1.inlet;
        flow : Fluid from valve1.outlet to fillingTank.inlet;

        // Flows - feeder path 2
        flow : Fluid from feederTank2.outlet to pump2.inlet;
        flow : Fluid from pump2.outlet to valve2.inlet;
        flow : Fluid from valve2.outlet to fillingTank.inlet;

        // System-level dynamics: tank levels change based on flow
        constraint feeder1DrainRate {
            feederTank1.outlet.flowRate == valve1.outlet.flowRate
        }

        constraint feeder2DrainRate {
            feederTank2.outlet.flowRate == valve2.outlet.flowRate
        }

        constraint fillingFillRate {
            fillingTank.inlet.flowRate == valve1.outlet.flowRate + valve2.outlet.flowRate
        }
    }

    part system : FillingSystem;
}
