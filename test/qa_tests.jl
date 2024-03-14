@testitem "Quality Assurance" setup=[SharedTestSetup] begin
    using Aqua, ExplicitImports

    Aqua.test_all(BatchedRoutines; ambiguities=false)
    Aqua.test_ambiguities(BatchedRoutines; broken=true)
    @test ExplicitImports.check_no_implicit_imports(BatchedRoutines) === nothing
end
