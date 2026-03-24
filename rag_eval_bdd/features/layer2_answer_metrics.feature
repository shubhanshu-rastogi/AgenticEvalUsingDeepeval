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

  @answer_relevancy @faithfulness @completeness @score_band_demo
  Scenario: Run layer2 answer metrics on demo score-band dataset
    Given backend is reachable
    And documents are uploaded from "eval/sample_docs/Match_Summary.pdf"
    And I load dataset "rag_eval_bdd/data/datasets/demo_score_band_questions.json"
    When I evaluate all questions
    Then save results for reporting

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

  @sanity3
  Scenario: Evaluate layer2 answer metrics from alternate inline dataset table
    Given backend is reachable
    And documents are uploaded from "eval/sample_docs/Match_Summary.pdf"
    And I use inline dataset:
      """
    | id   | question                                                                 | expected_answer                                                                                                              | category |
    | L2Q1 | Which bowler conceded 49 runs in his first two overs?                     | Anrich Nortje conceded 49 runs in his first two overs.                                                                        | bowling  |
    | L2Q2 | Who dropped Suryakumar Yadav, on which ball, and what shot did he attempt?| Corbin Bosch dropped Suryakumar Yadav on 10.2 when Suryakumar attempted a trademark scoop.                                    | fielding |
    | L2Q3 | How was Suryakumar Yadav eventually dismissed and who took the catch?     | Suryakumar Yadav was caught by Linde off Kwena Maphaka after getting a top-edge while trying to shovel the ball behind square.| wickets  |
    | L2Q4 | How did Tilak Varma score his runs during his innings?                    | Tilak Varma played aggressively, hitting multiple fours and sixes including consecutive big shots. | batting  |
    | L2Q5 | Describe the key moments in Suryakumar Yadav’s innings.                   | Suryakumar Yadav started with attacking shots including a first-ball six, survived a dropped catch on 10.2, and was eventually dismissed after top-edging a slower delivery. | batting  |
    | L2Q6 | How did Rinku Singh make an impact towards the end of the innings?        | Rinku Singh accelerated late in the innings by hitting boundaries including a six off Rabada after advancing down the track and creating scoring momentum. | batting  |  
      """
    When I evaluate all questions
    Then metric "answer_relevancy" should be >= configured threshold
    And metric "faithfulness" should be >= configured threshold
    And metric "completeness" should be >= configured threshold
    And save results for reporting

  @sanity5
  Scenario: Evaluate layer2 answer metrics from alternate inline dataset tables
    Given backend is reachable
    And documents are uploaded from "eval/sample_docs/Match_Summary.pdf"
    And I use inline dataset:
      """
    | id   | question                                                                 | expected_answer                                                                                                                                     | category |
    | L2Q1 | Which bowler conceded 49 runs in his first two overs?                    | Anrich Nortje conceded 49 runs in his first two overs and later dismissed Rinku Singh.                                                            | bowling  |
    | L2Q2 | Who dropped Suryakumar Yadav, on which ball, and what shot did he attempt? | Corbin Bosch dropped Suryakumar Yadav on 10.2 when Suryakumar attempted a trademark scoop.                                                       | fielding |
    | L2Q3 | How was Suryakumar Yadav eventually dismissed and who took the catch?    | Suryakumar Yadav was caught by Linde off Kwena Maphaka on 12.2 after top-edging a slower ball while trying to shovel it behind square.          | wickets  |
    | L2Q4 | How did Tilak Varma score his runs during his innings?                   | Tilak Varma scored 45 off 19 with 4 fours and 3 sixes, including aggressive boundary-hitting, before being bowled by Marco Jansen on 10.5.      | batting  |
    | L2Q5 | Describe the key moments in Suryakumar Yadav’s innings.                  | Suryakumar Yadav began with a first-ball six, survived a dropped catch by Bosch on 10.2 while attempting a trademark scoop, and was later dismissed for 30 off 16 after top-edging a slower ball to Linde off Kwena Maphaka. | batting  |
    | L2Q6 | How did Rinku Singh make an impact towards the end of the innings?       | Rinku Singh accelerated late in the innings by hitting boundaries including a six off Rabada after advancing down the track and creating scoring momentum. | batting  |
      """
    When I evaluate all questions
    Then metric "answer_relevancy" should be >= configured threshold
    And metric "faithfulness" should be >= configured threshold
    And metric "completeness" should be >= configured threshold
    And save results for reporting

  @demo
  Scenario: Evaluate layer2 answer metrics from alternate inline dataset table - sanity2
    Given backend is reachable
    And documents are uploaded from "eval/sample_docs/Match_Summary.pdf"
    And I use inline dataset:
      """
    | id   | question                                                                 | expected_answer                                                                                                                                               | category |
    | L2D3 | How did Hardik Pandya get out and on which delivery?                     | Hardik Pandya got out caught by Marco Jansen off the bowling of Corbin Bosch on the 19.5 delivery.                                                         | batting  |
    | L2D4 | Who scored maximum runs and how many boundaries ?                         | Ishan Kishan scored the maximum runs with a total of 50 runs. He hit 4 fours and 4 sixes during his innings.                                               | batting  |
    | L2D5 | What was the strike rate of Hardik Pandya and how did he score those runs? | Hardik Pandya scored 30 runs off 10 balls, which gives him a strike rate of 300. He achieved this by hitting 2 fours and 3 sixes during his innings. | batting  |
      """
    When I evaluate all questions
    Then metric "answer_relevancy" should be >= configured threshold
    And metric "faithfulness" should be >= configured threshold
    And metric "completeness" should be >= configured threshold
    And save results for reporting
