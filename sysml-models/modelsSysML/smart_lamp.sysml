package SmartLamp1 {
    private import ScalarValues::*;
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
        out item light : LightSignal;
    }

    port def ModePort {
        in item mode : ModeSignal;
    }

    port def ManualCommandPort {
        in item command : ManualCommandSignal;
    }

    part def MotionSensor {
        attribute motionDetected : Boolean;
        port sensorOut : ~MotionPort;

        action step {
            assign sensorOut.motion.detected := motionDetected;
        }
    }

    part def Lamp {
        attribute lampOn : Boolean;
        port lampIn : LightPort;

        state def LampStates {
            entry;
            then Off;

            state Off;
            state On;

            transition OffToOn
                first Off if lampIn.light.on do action { assign lampOn := true; } then On;

            transition OnToOff
                first On if not lampIn.light.on do action { assign lampOn := false; } then Off;
        }

        exhibit state lampStates : LampStates;
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

        action step {
            assign motionDetected := motionIn.motion.detected;
            assign autoMode := modeIn.mode.autoMode;
            assign manualOn := manualIn.command.manualOn;

            if autoMode and motionDetected {
                assign lampOn := true;
            }

            if autoMode and not motionDetected {
                assign lampOn := false;
            }

            if not autoMode {
                assign lampOn := manualOn;
            }

            assign lightOut.light.on := lampOn;
        }
    }

    part def ModeSelector {
        attribute autoMode : Boolean;
        port modeOut : ~ModePort;

        action step {
            assign modeOut.mode.autoMode := autoMode;
        }
    }

    part def UserPanel {
        attribute manualOn : Boolean;
        port manualOut : ~ManualCommandPort;

        action step {
            assign manualOut.command.manualOn := manualOn;
        }
    }

    part def SmartLampSystem {
        part sensor : MotionSensor;
        part modeSelector : ModeSelector;
        part userPanel : UserPanel;
        part controller : LampController;
        part lamp : Lamp;

        connect sensor.sensorOut to controller.motionIn;
        connect modeSelector.modeOut to controller.modeIn;
        connect userPanel.manualOut to controller.manualIn;
        connect controller.lightOut to lamp.lampIn;

        requirement def 'Lamp Follows Manual Command In Manual Mode' {
            doc
            /* When auto mode is off, the lamp should follow the manual command. */
            subject s : SmartLampSystem;

            require constraint {
                not s.controller.autoMode implies s.controller.lampOn == s.userPanel.manualOn
            }
        }

        requirement def 'Lamp Turns On In Auto Mode With Motion' {
            doc
            /* When auto mode is on and motion is detected, the lamp should be on. */
            subject s : SmartLampSystem;

            require constraint {
                (s.controller.autoMode and s.controller.motionDetected) implies s.controller.lampOn
            }
        }

        requirement def 'Lamp Turns Off In Auto Mode Without Motion' {
            doc
            /* When auto mode is on and no motion is detected, the lamp should be off. */
            subject s : SmartLampSystem;

            require constraint {
                (s.controller.autoMode and not s.controller.motionDetected) implies
                not s.controller.lampOn
            }
        }

        requirement def 'Controller Output Matches Lamp State' {
            doc
            /* The lamp state should match the controller output decision. */
            subject s : SmartLampSystem;

            require constraint {
                s.controller.lampOn == s.lamp.lampOn
            }
        }
    }

    part smartLamp : SmartLampSystem {
        part :>> sensor {
            attribute :>> motionDetected = false;
        }

        part :>> modeSelector {
            attribute :>> autoMode = true;
        }

        part :>> userPanel {
            attribute :>> manualOn = false;
        }

        part :>> controller {
            attribute :>> motionDetected = false;
            attribute :>> lampOn = false;
        }

        part :>> lamp {
            attribute :>> lampOn = false;
        }
    }

    part smartLampManual : SmartLampSystem {
        part :>> sensor {
            attribute :>> motionDetected = false;
        }

        part :>> modeSelector {
            attribute :>> autoMode = false;
        }

        part :>> userPanel {
            attribute :>> manualOn = true;
        }

        part :>> controller {
            attribute :>> motionDetected = false;
            attribute :>> lampOn = false;
        }

        part :>> lamp {
            attribute :>> lampOn = false;
        }
    }

    part smartLampAutoMotion : SmartLampSystem {
        part :>> sensor {
            attribute :>> motionDetected = true;
        }

        part :>> modeSelector {
            attribute :>> autoMode = true;
        }

        part :>> userPanel {
            attribute :>> manualOn = false;
        }

        part :>> controller {
            attribute :>> motionDetected = false;
            attribute :>> lampOn = false;
        }

        part :>> lamp {
            attribute :>> lampOn = false;
        }
    }

    part smartLampAutoNoMotion : SmartLampSystem {
        part :>> sensor {
            attribute :>> motionDetected = false;
        }

        part :>> modeSelector {
            attribute :>> autoMode = true;
        }

        part :>> userPanel {
            attribute :>> manualOn = true;
        }

        part :>> controller {
            attribute :>> motionDetected = false;
            attribute :>> lampOn = false;
        }

        part :>> lamp {
            attribute :>> lampOn = false;
        }
    }
}
