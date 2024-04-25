module BatchedRoutinesSciMLSensitivityExt

using ADTypes: AutoForwardDiff, AutoFiniteDiff
using BatchedRoutines: BatchedRoutines, BatchedNonlinearSolution
using FastClosures: @closure
using LinearSolve: LinearSolve
using SciMLBase: SciMLBase, NonlinearProblem, NonlinearSolution
using SciMLSensitivity: SciMLSensitivity, SteadyStateAdjoint, ZygoteVJP
using Zygote: Zygote

include("steadystateadjoint.jl")

end
