package SmartLamp {
    private import ScalarValues::*;

    metadata def Neural;
    metadata def NeuralRequirement;
    metadata def Prohibition;
    metadata def Obligation;
    metadata def ScenarioInput;
    metadata def ScenarioConstraint;

    item def MotionSignal {
        attribute detected : Boolean;
    }

    item def LightSignal {
        attribute on : Boolean;
    }

    item def ModeSignal {
        attribute autoMode : Boolean;
    }

    item def ManualCommandSignal {
        attribute manualOn : Boolean;
    }

    port def MotionPort {
        in item motion : MotionSignal;
    }

    port def LightPort {
        in item light : LightSignal;
    }

    port def ModePort {
        in item mode : ModeSignal;
    }

    port def ManualCommandPort {
        in item command : ManualCommandSignal;
    }

    part def MotionSensor {
        attribute motionDetectedCode : Natural;
        attribute motionDetected : Boolean;
        port sensorOut : ~MotionPort;

        action step {
            assign motionDetected := motionDetectedCode > 0;
            assign sensorOut.motion.detected := motionDetected;
        }
    }

    part def ModeSelector {
        attribute autoModeCode : Natural;
        attribute autoMode : Boolean;
        port modeOut : ~ModePort;

        action step {
            assign autoMode := autoModeCode > 0;
            assign modeOut.mode.autoMode := autoMode;
        }
    }

    part def UserPanel {
        attribute manualOnCode : Natural;
        attribute manualOn : Boolean;
        port manualOut : ~ManualCommandPort;

        action step {
            assign manualOn := manualOnCode > 0;
            assign manualOut.command.manualOn := manualOn;
        }
    }

    part def Lamp {
        attribute lampOn : Boolean;
        port lampIn : ~LightPort;

        action step {
            assign lampOn := lampIn.light.on;
        }
    }

    part def LampController {
        attribute motionDetected : Boolean;
        attribute lampOn : Boolean;
        attribute autoMode : Boolean;
        attribute manualOn : Boolean;

        port motionIn : MotionPort;
        port lightOut : ~LightPort;
        port modeIn : ModePort;
        port manualIn : ManualCommandPort;

        #Neural action def Policy {
            in motionDetected : Boolean;
            in autoMode : Boolean;
            in manualOn : Boolean;
            in isDone : Boolean;

            out lampState : Boolean;
        }

        #NeuralRequirement requirement def 'Neural Lamp Controller Soundness' {
            subject p : Policy;
            require constraint {
                ((p.autoMode implies (p.lampState == p.motionDetected)) and
                 ((not p.autoMode) implies (p.lampState == p.manualOn)))
            }
        }

        state def ControllerStates {
            entry;
            then AutoState;

            state AutoState;
            state ManualState;

            transition AutoToManual
                first AutoState
                if not autoMode
                then ManualState;

            transition ManualToAuto
                first ManualState
                if autoMode
                then AutoState;
        }

        exhibit state controllerStates : ControllerStates;

        action step {
            assign motionDetected := motionIn.motion.detected;
            assign autoMode := modeIn.mode.autoMode;
            assign manualOn := manualIn.command.manualOn;

            action policyCall : Policy {
                in motionDetected := motionDetected;
                in autoMode := autoMode;
                in manualOn := manualOn;
                in isDone := false;
            }

            assign lampOn := policyCall.lampState;
            assign lightOut.light.on := lampOn;
        }
    }

    part def SmartLampSystem {
    part sensor : MotionSensor;
    part modeSelector : ModeSelector;
    part userPanel : UserPanel;
    part controller : LampController;
    part lamp : Lamp;

    flow from sensor.sensorOut to controller.motionIn;
    flow from modeSelector.modeOut to controller.modeIn;
    flow from userPanel.manualOut to controller.manualIn;
    flow from controller.lightOut to lamp.lampIn;

    #Obligation requirement def 'Manual Off Keeps Lamp Off' {
        subject s : SmartLampSystem;
        require constraint {
            (((not s.controller.autoMode) and (not s.controller.manualOn)) implies
             (not s.controller.lampOn))
        }
    }

    #Obligation requirement def 'Auto Mode Without Motion Keeps Lamp Off' {
        subject s : SmartLampSystem;
        require constraint {
            (((s.controller.autoMode) and (not s.controller.motionDetected)) implies
             (not s.controller.lampOn))
        }
    }

    #Obligation requirement def 'Auto Mode Lamp On Requires Motion' {
        subject s : SmartLampSystem;
        require constraint {
            (((s.controller.autoMode) and s.controller.lampOn) implies
             s.controller.motionDetected)
        }
    }

    #Prohibition requirement def 'Lamp On Must Have Valid Cause' {
        subject s : SmartLampSystem;
        require constraint {
            (s.controller.lampOn implies
             (((s.controller.autoMode and s.controller.motionDetected) or
               ((not s.controller.autoMode) and s.controller.manualOn))))
        }
    }
}



    part system : SmartLampSystem {
        part :>> sensor {
            #ScenarioInput attribute :>> motionDetectedCode = 0;
            attribute :>> motionDetected = false;
        }

        part :>> modeSelector {
            #ScenarioInput attribute :>> autoModeCode = 1;
            attribute :>> autoMode = true;
        }

        part :>> userPanel {
            #ScenarioInput attribute :>> manualOnCode = 0;
            attribute :>> manualOn = false;
        }

        part :>> controller {
            attribute :>> motionDetected = false;
            attribute :>> lampOn = false;
            attribute :>> autoMode = true;
            attribute :>> manualOn = false;
        }

        part :>> lamp {
            attribute :>> lampOn = false;
        }

        #ScenarioConstraint constraint scenarioConstraint {
            sensor.motionDetectedCode >= 0 and sensor.motionDetectedCode <= 1 and
            modeSelector.autoModeCode >= 0 and modeSelector.autoModeCode <= 1 and
            userPanel.manualOnCode >= 0 and userPanel.manualOnCode <= 1
        }
    }
}
