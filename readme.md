### Cellular Automata

Cellular Automata (CA) is a mathematical model that is used to simulate the behavior of systems composed of a large number of simple, identical units called "cells." Each cell is assigned a state, which can be represented as a binary value (0 or 1), a color, or a number. The state of each cell evolves over time based on a set of rules that determine the new state of a cell based on the current state of itself and its neighboring cells. The rules are applied to all cells simultaneously, and the process is repeated in a time-stepping manner, resulting in the evolution of the entire system.

Cellular automata have been used in a variety of fields, including computer science, physics, mathematics, and biology, to study complex systems and emergent phenomena. Some well-known examples of cellular automata include John Horton Conway's "Game of Life," a simulation of cellular evolution in two-dimensional space, and Wolfram's "Class 1" and "Class 4" automata, which display complex, seemingly random behavior.



### Lenia
"Lenia and Expanded Universe" by Chan, Bert Wang-Chak (2020) is a research paper that presents Lenia, an artificial life platform that simulates ecosystems of digital creatures. Lenia's ecosystem consists of simple, artificial organisms that can move, consume, grow, and reproduce. The paper explains how Lenia's system is designed to be scalable and flexible, allowing for the creation of a diverse range of organisms with varying abilities and behaviors. The authors also discuss the idea of an "expanded universe" for Lenia, in which multiple Lenia systems can be linked together to form a larger, interconnected ecosystem. The paper concludes by highlighting the potential applications and implications of Lenia as a tool for studying artificial life and evolution.


Some random behavior generrated by fixing python3 Lenia.py --mu 0.49 --sigma 0.275

![](https://github.com/s4nyam/APCSP/blob/main/current_version_code/outputs/output.gif)


### Projct Description

#### Introduction
In the 1950s, Jon Von Neuman and his colleague Ulam proposed an idea of self-reproduction in machines using CA-like rules which were further formalized by Wolfram. Simple agents interact with their neighbors and decide whether they are going to stay alive, die or give birth to a new cell. This behavior is observed in 1-D CA and 2-D CA based on their geographical and temporal instantiation of cells. Each cell has two properties (at least) – Current state and Neighborhood. Mathematically,
<img width="378" alt="image" src="https://user-images.githubusercontent.com/13884479/218536452-d99275de-7a55-4425-92e2-372e02123719.png">

We target two models of Cellular Automata (CA) and investigate emergent dynamics by identifying metrics to measure such behavior. Further, by the implementation of CAs, we intend to propose updated rules that identify the next state of the CA. For example, in 2D CA, most of the behaviors are way boring. However, some sets of rules act as a magical sweet spot where life-like rules start to emerge, for example, GoL. 


#### Research Question
Can we find and measure long-term complex emerging behavior of rules in different CA models, for example, Multi-State CA, Continuous CA, or Neural CA?

#### Objectives

Our primary and secondary objectives are:
-	Primary: Measuring the long-term complexity with a known scheme of emerging behavior in CA. Also, to find updated rules that could predict the next state of the CA grid.
-	Secondary: Identifying and comparing other schemes for 1. Measuring complexity 2. Update rules


#### Literature Review
CA has been studied widely and timely throughout. We particularly focus on producing long-term dynamics with different methods:

-	[1] A neural CA has been proposed. In this work, the authors really hope that the individual organisms that evolve within the environment may somehow learn the dynamics of interacting with each other. Each agent-like behavior in the environment has its own head, neural operation, and learning capability. This makes it possible to work like a colony where every individual is working for the colony and propagating itself. Authors emphasize the capabilities of the individuals in the environment to be self-healing, organizing, and learning underlying nature’s law and physical interpretations that make them open-ended. In short, contrary to the use of CA rules, general NN is embedded in cells that can together become a larger NN, where: 1. Cells can clone themselves. 2. Communicate with each other. 3. Control the physical and chemical nature of the grid.

-	The paper [2] proposes a new model HetCA, “heterogeneous cellular automata”, that is, the extension of CA to study open-ended evolution. The model diverges from classical cellular automata in three ways: firstly, cells have properties of decay and quiescence. Secondly, each cell contains its own transition function, also called the genome, which can evolve over time and be transferred to its neighbors, and thirdly, these genomic transition functions are represented in a parsimonious way. The outcome of the combination of these changes leads to the formation of an evolving ecosystem where different colonies of cells compete with one another.  The results showed that HetCA was capable of long-term phenotypic dynamics and sustained a high level of variance over long runs. Furthermore, the model displayed greater behavioral diversity than classical cellular automata, such as the Game of Life. 



The detailed workflow has been given in Figure (1) represented by Gantt Chart. The project has been identified with Main Tasks and the subtasks, with touch base with the supervisor for continuous inputs for successful completion of the project.

<img width="454" alt="image" src="https://user-images.githubusercontent.com/13884479/218536541-ea413bb2-e614-42aa-bcdc-43e8b720d9d2.png">

[1] Gregor, K., & Besse, F. (2021). Self-organizing intelligent matter: A blueprint for an ai generating algorithm. arXiv preprint arXiv:2101.07627.

[2] Medernach, D., Kowaliw, T., Ryan, C., & Doursat, R. (2013, July). Long-term evolutionary dynamics in heterogeneous cellular automata. In Proceedings of the 15th annual conference on Genetic and evolutionary computation (pp. 231-238).





### References

- Chan, Bert Wang-Chak (2020). Lenia and Expanded Universe. Artificial Life Conference Proceedings, (32), 221–229. arXiv:2005.03742.

- Chan, Bert Wang-Chak (2019). Lenia: Biology of Artificial Life. Complex Systems, 28(3), 251–286. arXiv:1812.05433.

- Lenia LH Github repo - https://github.com/ljhowell/lenia-lh



###The project is part of the course in Applied Computer Science Project at Østfold University College (Norway), under the supervision of Stefano Nichele (nichele.eu)
