@layer2
Feature: Layer 2 answer quality evaluation

  @sanity @smoke @answer_relevancy
  Scenario: Smoke layer2 answer relevancy with one inline row
    Given backend is reachable
    And documents are uploaded from "eval/sample_docs/Match_Summary.pdf"
    And I use inline dataset:
      """
      | id   | question                             | expected_answer           | category |
      | L2S1 | How many sixes did Tilak Varma hit? | Tilak Varma hit 3 sixes.  | batting  |
      """
    When I evaluate all questions with metrics "answer_relevancy"
    Then metric "answer_relevancy" should be >= configured threshold
    And save results for reporting

  @sanity @answer_relevancy @faithfulness @completeness @regression
  Scenario: Evaluate layer2 answer metrics from inline dataset table
    Given backend is reachable
    And documents are uploaded from "eval/sample_docs/Match_Summary.pdf"
    And I use inline dataset:
      """
      | id   | question                                                           | expected_answer                                                                 | category |
      | L2Q1 | How many sixes did Tilak Varma hit?                               | Tilak Varma hit 3 sixes.                                                        | batting  |
      | L2Q2 | On which delivery was Tilak Varma dismissed and how?              | Tilak Varma was dismissed on the 11th delivery by Marco Jansen, bowled behind his legs. | wickets  |
      | L2Q3 | Which bowler conceded 49 runs in two overs and what key events happened? | Nortje conceded 49 runs in his first two overs and later dismissed Rinku Singh. | bowling  |
      """
    When I evaluate all questions
    Then metric "answer_relevancy" should be >= configured threshold
    And metric "faithfulness" should be >= configured threshold
    And metric "completeness" should be >= configured threshold
    And save results for reporting

  @answer_relevancy @faithfulness @completeness
  Scenario: Evaluate layer2 answer metrics from external dataset file
    Given backend is reachable
    And documents are uploaded from "eval/sample_docs/Match_Summary.pdf"
    And I load dataset "rag_eval_bdd/data/datasets/layer2_questions.json"
    When I evaluate all questions
    Then metric "answer_relevancy" should be >= configured threshold
    And metric "faithfulness" should be >= configured threshold
    And metric "completeness" should be >= configured threshold
    And save results for reporting

  @live @answer_relevancy @faithfulness @completeness
  Scenario: Evaluate layer2 answer metrics on runtime live dataset
    Given backend is reachable
    And I use latest uploaded session from application UI
    And I generate live dataset for layer "layer2" from uploaded documents
    When I evaluate all questions
    Then metric "answer_relevancy" should be >= configured threshold
    And metric "faithfulness" should be >= configured threshold
    And metric "completeness" should be >= configured threshold
    And save results for reporting
  @sanity1
  Scenario: Evaluate layer2 answer metrics from alternate inline dataset table
    Given backend is reachable
    And documents are uploaded from "eval/sample_docs/Match_Summary.pdf"
    And I use inline dataset:
      """
      | id   | question                                                                 | expected_answer                                                                                                 | category |
      | M1Q4 | Who dropped Suryakumar Yadav, on which ball, and what shot did he attempt? | Corbin Bosch dropped Suryakumar Yadav on 10.2 when Suryakumar attempted a trademark scoop.                     | wickets  |
      | M1Q5 | How did Hardik Pandya get out, and who took the catch?                    | Hardik Pandya was caught off Corbin Bosch, and Marco Jansen took the catch at sweeper cover.                   | wickets  |
      | M1Q6 | Which boundary-scoring shots by Axar Patel are mentioned in the summary?  | Axar Patel hit Linde for a six at 11.4, Rabada for a four at 14.3, Rabada for a six at 14.4, and Rabada for a four at 18.3. | batting  |
      """
    When I evaluate all questions
    Then metric "answer_relevancy" should be >= configured threshold
    And metric "faithfulness" should be >= configured threshold
    And metric "completeness" should be >= configured threshold
    And save results for reporting
