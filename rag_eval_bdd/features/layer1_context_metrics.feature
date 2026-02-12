@layer1
Feature: Layer 1 retrieval quality evaluation

  @sanity @smoke @contextual_precision
  Scenario: Smoke layer1 contextual precision with one inline row
    Given backend is reachable
    And documents are uploaded from "eval/sample_docs/Match_Summary.pdf"
    And I use inline dataset:
      """
      | id   | question                             | expected_answer           | category |
      | L1S1 | How many sixes did Tilak Varma hit? | Tilak Varma hit 3 sixes.  | batting  |
      """
    When I evaluate all questions with metrics "contextual_precision"
    Then metric "contextual_precision" should be >= configured threshold
    And save results for reporting

  @sanity @contextual_precision @contextual_recall @contextual_relevancy
  Scenario: Evaluate layer1 contextual metrics from inline dataset table
    Given backend is reachable
    And documents are uploaded from "eval/sample_docs/Match_Summary.pdf"
    And I use inline dataset:
      """
      | id   | question                                              | expected_answer                                | category |
      | L1Q1 | How many sixes did Tilak Varma hit?                  | Tilak Varma hit 3 sixes.                       | batting  |
      | L1Q2 | Who dismissed Suryakumar Yadav?                      | Suryakumar Yadav was dismissed by Maphaka.     | wickets  |
      | L1Q3 | Which bowler conceded 49 runs in two overs and what key events happened during that spell? | Nortje conceded 49 runs in his first two overs and later dismissed Rinku Singh, caught by Stubbs. | bowling |
      """
    When I evaluate all questions
    Then metric "contextual_precision" should be >= configured threshold
    And metric "contextual_recall" should be >= configured threshold
    And metric "contextual_relevancy" should be >= configured threshold
    Then save results for reporting

  @regression @contextual_precision @contextual_recall @contextual_relevancy
  Scenario: Evaluate layer1 contextual metrics from external dataset file
    Given backend is reachable
    And documents are uploaded from "eval/sample_docs/Match_Summary.pdf"
    And I load dataset "rag_eval_bdd/data/datasets/layer1_questions.json"
    When I evaluate all questions
    Then metric "contextual_precision" should be >= configured threshold
    And metric "contextual_recall" should be >= configured threshold
    And metric "contextual_relevancy" should be >= configured threshold
    And save results for reporting
