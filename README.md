# Development Notes
## MOTAP Framework Project Layout

src /
    |- agent /
            |- agent.rs: Contains MDP and Team (team of agents) implementation
    |- algorithm /
            |- dp.rs: Contains dynamic programming functions: Value iteration, Value iteration for initial scheduler, random proper scheduler generation, argmax transition and reward matrix creation
            |- synth.rs: Scheduler synthesis algorithm
    |- c-binding /
            |- suite_sparse.rs: FFI for CX Sparse from Suite-Sparse - sparse matrix BLAS functions
    |- dfa /
            |- dfa.rs: deterministic finite automaton, and Mission (batch of tasks) implementation
    |- lp /
            |- pylp.py: python GIL linear programming interface. Rust calls these scripts
            through py03. Written in python to fill the scientific computing gap in Rust.
    |- parallel /
            |- threaded.rs: processing a collection of multi-objective product MDPs instatiated using MOProductMDP struct and implementation. Generates an mpsc channel, and a fixed size threadpool to compute the value iteration of the MDPs in parallel. 
    |- scpm / (Could be deprecated)
            |-model.rs: 
    
