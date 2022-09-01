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

            |-model.rs: Contains the Product MDP M x A where M is the agent MDP and A is a DFA
             corresponding to task j, called the MOProductMDP and its implementation. Also Contains 
             a MOTAP problem meta struct called SCPM, which for a given batch, contains the team of agents -> Team and the collection of tasks -> Mission. SCPM implements the product 
             builder function for generating MOProductMDPs

    |- lib.rs: Library of general helper functions and utilities. Contains cBLAS FFI functions, the wrappert to CX Sparse BLAS functions, value iteration helpers, python interface linking, numeric helpers, and lp python wrappers

tests/

    |- testing the framework interface on a simple scalable problem

build.rs: Contains the build file for linking to C libs. 

example.py: an example file constructing the event caller for a continuous stream of tasks which is sent over a redis channel (setup to be local but can be edited for server based implementation). Also contains a task executer for executing schedules sent back from the batch solver in pipeline.py

pipeline.py contains the event listener and constructs batches of tasks to be bundled into a MOTAP problem. It sends a randomised scheduler to be rendered by a collection of agents. 
    
