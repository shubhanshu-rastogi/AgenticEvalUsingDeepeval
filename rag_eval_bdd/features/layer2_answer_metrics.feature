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

  @sanity @answer_relevancy @faithfulness @completeness
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

  @regression @answer_relevancy @faithfulness @completeness
  Scenario: Evaluate layer2 answer metrics from external dataset file
    Given backend is reachable
    And documents are uploaded from "eval/sample_docs/Match_Summary.pdf"
    And I load dataset "rag_eval_bdd/data/datasets/layer2_questions.json"
    When I evaluate all questions
    Then metric "answer_relevancy" should be >= configured threshold
    And metric "faithfulness" should be >= configured threshold
    And metric "completeness" should be >= configured threshold
    And save results for reporting
