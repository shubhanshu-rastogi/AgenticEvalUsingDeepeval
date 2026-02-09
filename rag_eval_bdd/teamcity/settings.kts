import jetbrains.buildServer.configs.kotlin.v2019_2.*
import jetbrains.buildServer.configs.kotlin.v2019_2.buildSteps.script

version = "2022.10"

project {
    buildType(RagEvalBdd)
}

object RagEvalBdd : BuildType({
    name = "RAG Eval BDD"

    vcs {
        root(DslContext.settingsRoot)
    }

    steps {
        script {
            name = "Setup Python"
            scriptContent = """
                cd rag_eval_bdd
                python3 -m venv .venv-ci
                . .venv-ci/bin/activate
                pip install --upgrade pip
                pip install -r requirements.txt
            """.trimIndent()
        }

        script {
            name = "Smoke Gate"
            scriptContent = """
                cd rag_eval_bdd
                . .venv-ci/bin/activate
                pytest -c pytest.ini tests -m "smoke" -q
            """.trimIndent()
        }

        script {
            name = "Run Live BDD (Non-blocking)"
            scriptContent = """
                cd rag_eval_bdd
                . .venv-ci/bin/activate
                pytest -c pytest.ini steps -m "live and (layer1 or layer2)" --alluredir=allure-results || true
            """.trimIndent()
        }
    }

    artifactRules = """
        rag_eval_bdd/allure-results => allure-results
        rag_eval_bdd/allure-report => allure-report
        rag_eval_bdd/results => eval-results
    """.trimIndent()
})
