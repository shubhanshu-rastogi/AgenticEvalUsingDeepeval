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

  @contextual_precision @contextual_recall @contextual_relevancy  @score_band_demo
  Scenario: Run layer1 contextual metrics on demo score-band dataset
    Given backend is reachable
    And documents are uploaded from "eval/sample_docs/Match_Summary.pdf"
    And I load dataset "rag_eval_bdd/data/datasets/demo_score_band_questions.json"
    When I evaluate all questions
    Then save results for reporting

  @live @contextual_precision @contextual_recall @contextual_relevancy
  Scenario: Evaluate layer1 contextual metrics on runtime live dataset
    Given backend is reachable
    And I use latest uploaded session from application UI
    And I generate live dataset for layer "layer1" from uploaded documents
    When I evaluate all questions
    Then metric "contextual_precision" should be >= configured threshold
    And metric "contextual_recall" should be >= configured threshold
    And metric "contextual_relevancy" should be >= configured threshold
    And save results for reporting
  @sanity1
  Scenario: Evaluate layer1 contextual metrics from alternate inline dataset table
    Given backend is reachable
    And documents are uploaded from "eval/sample_docs/Match_Summary.pdf"
    And I use inline dataset:
      """
      | id   | question                                                                 | expected_answer                                                                                                 | category |
      | M1Q1 | Which bowler was hit for 49 runs in his first two overs, and which batter later got out to him? | Anrich Nortje conceded 49 runs in his first two overs, and later Rinku Singh was dismissed off his bowling.    | bowling  |
      | M1Q2 | How was Tilak Varma dismissed, on which ball, and who was the bowler?    | Tilak Varma was bowled by Marco Jansen on 11.0, and he was dismissed behind his legs.                          | wickets  |
      | M1Q3 | What happened at 5.4, and why was that moment notable?                    | At 5.4, Ishan Kishan hit Kwena Maphaka for six, brought up his fifty off 23 balls, and then retired out.      | batting  |
      """
    When I evaluate all questions
    Then metric "contextual_precision" should be >= configured threshold
    And metric "contextual_recall" should be >= configured threshold
    And metric "contextual_relevancy" should be >= configured threshold
    Then save results for reporting


  @sanity2
  Scenario: Evaluate layer1 contextual metrics from alternate inline dataset table
    Given backend is reachable
    And documents are uploaded from "eval/sample_docs/Match_Summary.pdf"
    And I use inline dataset:
    """
    | id   | question                                                                 | expected_answer                                                                 | category |
    | M1Q2 | On which delivery was Suryakumar Yadav dropped and which fielder missed it? | Suryakumar Yadav was dropped on 10.2 by Corbin Bosch.                            | fielding |
    | M1Q3 | Who dismissed Suryakumar Yadav and which fielder completed the catch?     | Suryakumar Yadav was dismissed by Kwena Maphaka and caught by Linde.             | wickets  |
    """
    When I evaluate all questions
    Then metric "contextual_precision" should be >= configured threshold
    And metric "contextual_recall" should be >= configured threshold
    And metric "contextual_relevancy" should be >= configured threshold
    Then save results for reporting
