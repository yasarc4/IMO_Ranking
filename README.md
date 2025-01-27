# IMO Ranking Contest - EDA

The below figures compares the features before and after transformation. Before transformation, many inter-correlated features were found to be important variables. When transformed based on EDA, the results were better and the dependent variables were only mildly correlated.

### Distribution of Mean Rank(target)
![Mean Rank Distribution](https://raw.githubusercontent.com/yasarc4/IMO_Ranking/master/Plots/MeanRank%20Distribution.png)

## Results before transformation
#### Relation between Raw scores and Mean Rank(Target)
 * Correlation above 0.6
 ![Correlations above 0.6 on Raw Scores](https://raw.githubusercontent.com/yasarc4/IMO_Ranking/master/Plots/above_60_direct.png)
 * Correlation above 0.55
 ![Correlations above 0.55 on Raw Scores](https://raw.githubusercontent.com/yasarc4/IMO_Ranking/master/Plots/above_55_direct.png)
 * Correlation above 0.5
 ![Correlations above 0.5 on Raw Scores](https://raw.githubusercontent.com/yasarc4/IMO_Ranking/master/Plots/above_50_direct.png)
 * Correlations between top 20 features
 ![Top 20 Raw Features - Correlation](https://raw.githubusercontent.com/yasarc4/IMO_Ranking/master/Plots/top_20_features_direct.png)
 * Top 20 features of the top 5 countries against mean rank
 ![Distribution of raw scores for top 5 countries](https://raw.githubusercontent.com/yasarc4/IMO_Ranking/master/Plots/Top_Countries_Scores.png)
 ![Raw Scores of Top 5 Countries GIF](https://raw.githubusercontent.com/yasarc4/IMO_Ranking/master/Plots/Top_Features_scores.gif)
 * Top 20 features of the next 10 countries against mean rank
 ![Distribution of raw scores for next 10 countries](https://raw.githubusercontent.com/yasarc4/IMO_Ranking/master/Plots/Top_Countries_Scores2.png)
 ![Raw Scores of Next 10 Countries GIF](https://raw.githubusercontent.com/yasarc4/IMO_Ranking/master/Plots/Top_Features2_scores.gif)

## Results after transformation
### Rank Distribution of various indicators
 * #### Top 5 Countries
 ![Distribution of all ranks in top 5 countries](https://raw.githubusercontent.com/yasarc4/IMO_Ranking/master/Plots/Rank%20Distribution%20Top%205.png)
 * #### Next 10 Countries
 ![Distribution of all ranks in next 10 countries](https://raw.githubusercontent.com/yasarc4/IMO_Ranking/master/Plots/Rank%20Distribution%20Next%2010.png)
#### Relation between Ranks(derived from score) and Mean Rank(Target)
 * Correlation above 0.6
 ![Correlations above 0.6 on Derived Features](https://raw.githubusercontent.com/yasarc4/IMO_Ranking/master/Plots/above_60.png)
 * Correlation above 0.55
 ![Correlations above 0.55 on Derived Features](https://raw.githubusercontent.com/yasarc4/IMO_Ranking/master/Plots/above_55.png)
 * Correlation above 0.5
 ![Correlations above 0.5 on Derived Features](https://raw.githubusercontent.com/yasarc4/IMO_Ranking/master/Plots/above_50.png)
 * Correlations between top 20 features
 ![Top 20 Derived Features - Correlation](https://raw.githubusercontent.com/yasarc4/IMO_Ranking/master/Plots/Top_20_features_correlations.png)
 * Top 20 features of the top 5 countries against mean rank
 ![Distribution of ranks for top 5 countries](https://raw.githubusercontent.com/yasarc4/IMO_Ranking/master/Plots/Top_Countries_Ranking.png)
 ![Ranks of Top 5 Countries GIF](https://raw.githubusercontent.com/yasarc4/IMO_Ranking/master/Plots/Top_Features.gif)
 * Top 20 features of the next 10 countries against mean rank
 ![Distribution of ranks for next 10 countries](https://raw.githubusercontent.com/yasarc4/IMO_Ranking/master/Plots/Top_countries_ranking2.png)
 ![Ranks of next 10 Countries GIF](https://raw.githubusercontent.com/yasarc4/IMO_Ranking/master/Plots/Top_Features2.gif)

 > All the above EDA done in 3 hrs for a hackathon.
