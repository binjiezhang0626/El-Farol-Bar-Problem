# El-Farol-Bar-Problem
The El Farol Bar problem is a classical game-theoretic problem studied in economics. In Santa Fe, there is a bar called El Farol. Every Thursday night, the bar organises an event which everyone in the town wants to attend. If less than 60 % of the population attends the bar, all have a better time in the bar than staying home. However, if 60 % or more of the population attends the bar, it becomes too crowded, and it would have been better for everyone to stay at home. We assume that everyone decides individually whether to go to the bar without communicating with anyone. However, the attendance in the bar during all past weeks is known to everyone.

We will model the scenario using co-evolution. We will represent the strategies of each player with a state-based representation S = (p, A, B), where p = (p 1 , . . . , p h ) is a vector of “attendance” probabilities, A = (a ij ) is a state transition matrix in case the bar is crowded, and B = (b ij ) is a state transition matrix in case the bar is not crowded.

In any week t, the individual is in one of h states. If the individual is in state i in week t, then she goes to the bar with probability p i that week. If the bar was crowded in week t, then in the following week, she transitions to state j with probability a ij . If the bar was not crowded in week t, then she transitions to state j with probability b ij in week t + 1. The individuals in the population all start in state 0, but have diﬀerent strategies.

This hypothetical problem models many real-world economic scenarios. We are interested in understanding the conditions under which the population can achieve an eﬃcient solution, i.e., a solution where the utilisation of a limited resource (in this case the bar) is as close as possible to its capacity. The problem is interesting, because the population cannot use the resource eﬃciently if all individuals use the same deterministic strategy.
