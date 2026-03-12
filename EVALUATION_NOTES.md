- [Austin] Systems:
    - Mixing model
    - Thermostat
    - Cruise controller

- [] Table:
    Columns:
        - System under evaluation
        - Number of requirements verified 
        - nuXmv performance characteristics
            - Runtime 
            - Memory
        - Heuristics: Count the number of applications of each heuristic
    Last row: A summary

- [] How much are we automating?
    - How many things are we extracting from the SysML file versus how much longer is the SysML file?
        - How long is the extracted SMV code from the SysML file? (LOC)
        - Reward function (LOC)

- [] Scaling argument - Graph with time in the Y axis, the X axis scales in size of SysML model (LOC)
    - [] Extraction runtime
    - [] nuXmv runtime
    - [] RL step

- [] Runtime monitoring -
    - Effective: generate synthetic data from a neural controller and show the runtime monitor
                 detects requirement violations. 
                 Table - one row per system. One column is the number of invariant violations, and the other column
                 is the number of detected invariant violations. 
    - Performance: runtime (ms), memory
        Chart: The X axis is the size of the requirements and the Y axis is runtime and memory (?)
