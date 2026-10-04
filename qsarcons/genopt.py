import random
from collections import Counter
from operator import attrgetter
from typing import Callable, Iterator, List


class Individual:
    """A candidate solution: an unordered set of unique gene values (e.g. model indices in a consensus)."""

    def __init__(self, genes: List[int]) -> None:
        self.genes = list(genes)
        self.score = 0.0

    def __eq__(self, other: "Individual") -> bool:
        """Two individuals are equal if they contain the same set of genes."""
        return sorted(self.genes) == sorted(other.genes)

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.genes)))

    def __repr__(self) -> str:
        return repr(sorted(self.genes))


class Population:
    """A collection of individuals, with sorting support."""

    def __init__(self) -> None:
        self.individuals: List[Individual] = []

    def __len__(self) -> int:
        return len(self.individuals)

    def __iter__(self) -> Iterator[Individual]:
        return iter(self.individuals)

    def append(self, individual: Individual) -> None:
        self.individuals.append(individual)

    def sort(self) -> "Population":
        """Sort individuals by score, best (highest) first."""
        self.individuals.sort(key=attrgetter("score"), reverse=True)
        return self

    def best_score(self) -> Individual:
        """Return the individual with the highest score."""
        best = max(self, key=attrgetter("score"))
        return best


class GeneticAlgorithm:
    """Genetic algorithm searching for the fixed-size subset of unique gene indices maximizing a fitness function."""

    def __init__(
        self,
        fitness_func: Callable[[Individual], float],
        n_genes: int,
        ind_size: int,
        pop_size: int = 50,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.2,
        random_seed: int = 42,
        verbose: bool = False,
    ) -> None:
        if ind_size > n_genes:
            raise ValueError(f"ind_size ({ind_size}) cannot exceed n_genes ({n_genes}).")

        self.fitness_func = fitness_func
        self.n_genes = n_genes
        self.ind_size = ind_size
        self.pop_size = pop_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.verbose = verbose

        self.rng = random.Random(random_seed)
        self.current_generation = 0

        self.population = init_population(self.pop_size, self.n_genes, self.ind_size, self.rng)
        self._evaluate(self.population)
        self.population.sort()

        self.best_solution = self.population.best_score()
        self.best_individuals = [self.best_solution]

        self.gene_counts = Counter()
        self.total_individuals = 0
        self._update_gene_counts()

    def __repr__(self) -> str:
        return f"<GeneticAlgorithm gen={self.current_generation} pop_size={self.pop_size}>"

    def _evaluate(self, population: Population) -> None:
        """Score every individual in a population using the fitness function."""
        for ind in population:
            ind.score = self.fitness_func(ind)

    def _update_gene_counts(self) -> None:
        """Accumulate how often each gene appears in the population, cumulatively across all generations."""
        for ind in self.population:
            self.gene_counts.update(ind.genes)
        self.total_individuals += len(self.population)

    def _make_individual(self, parent_a: Individual, parent_b: Individual, new_population: Population) -> Individual:
        """Create one individual via crossover + mutation, retrying on collision with new_population."""
        max_retries = 20
        child = parent_a
        for _ in range(max_retries):
            do_crossover = self.rng.random() <= self.crossover_prob
            base = shuffle_crossover(parent_a, parent_b, self.rng) if do_crossover else parent_a
            child = uniform_mutation(base, self.n_genes, self.mutation_prob, self.rng)
            if child not in new_population:
                break
        return child

    def step(self) -> "GeneticAlgorithm":
        """Advance the algorithm by one generation."""

        # 1. Selection: build a mating pool via tournament selection
        new_population = Population()
        mating_pool = tournament_selection(self.population, self.rng)

        # 2. Crossover + mutation: create two children from each mating pair
        num_pairs = self.pop_size // 2
        for _ in range(num_pairs):
            mother = mating_pool.pop(self.rng.randrange(len(mating_pool)))
            father = mating_pool.pop(self.rng.randrange(len(mating_pool)))

            new_population.append(self._make_individual(mother, father, new_population))
            new_population.append(self._make_individual(father, mother, new_population))

        # 3. Carry over the leftover parent when pop_size is odd
        if len(new_population) < self.pop_size:
            new_population.append(mating_pool.pop())

        # 4. Evaluate and sort the new generation
        self._evaluate(new_population)
        new_population.sort()

        # 5. Elitism: keep the best solution ever found alive in the population
        if self.best_solution.score > new_population.best_score().score:
            new_population.individuals[-1] = self.best_solution
            new_population.sort()
        else:
            self.best_solution = new_population.best_score()
        self.best_individuals.append(self.best_solution)

        # 6. Replace the current population and advance the generation counter
        self.population = new_population
        self.current_generation += 1
        self._update_gene_counts()
        return self

    def run(self, n_iter: int = 50) -> "GeneticAlgorithm":
        """Run the algorithm for a fixed number of generations."""
        for _ in range(n_iter):
            self.step()
            if self.verbose:
                self.report()
        return self

    def calc_stat(self) -> dict:
        """Compute this generation's participation and the run's cumulative popularity statistics."""
        current_genes = set()
        for ind in self.population:
            current_genes.update(ind.genes)

        popularity = [(gene, count / self.total_individuals * 100) for gene, count in self.gene_counts.most_common(5)]

        stats = {
            "n_participated": len(current_genes),
            "n_total": self.n_genes,
            "n_never_participated": self.n_genes - len(self.gene_counts),
            "popularity": popularity,
        }
        return stats

    def report(self) -> None:
        """Print generation number, best score, model-participation counts, and top model popularity."""
        stats = self.calc_stat()
        popularity = "/".join(f"{gene}[{pct:.1f}%]" for gene, pct in stats["popularity"])
        participation = f"{stats['n_participated']}/{stats['n_total']}/{stats['n_never_participated']}"
        print(
            f"Gen {self.current_generation} | "
            f"Best score: {self.best_solution.score:.3f} | "
            f"Models participated/total/remained: {participation} | "
            f"Model popularity: {popularity}"
        )

    def get_solution(self) -> Individual:
        """Return the best individual found across all generations."""
        solution = max(self.best_individuals, key=attrgetter("score"))
        return solution


def init_individual(n_genes: int, ind_size: int, rng: random.Random) -> Individual:
    """Create a random individual: ind_size unique genes drawn from range(n_genes)."""
    individual = Individual(rng.sample(range(n_genes), k=ind_size))
    return individual


def init_population(pop_size: int, n_genes: int, ind_size: int, rng: random.Random) -> Population:
    """Create a population of unique random individuals."""
    population = Population()
    while len(population) < pop_size:
        candidate = init_individual(n_genes, ind_size, rng)
        if candidate not in population:
            population.append(candidate)
    return population


def shuffle_crossover(mother: Individual, father: Individual, rng: random.Random) -> Individual:
    """Child is a random sample from the parents' combined gene pool (always valid, no duplicates)."""
    pool = list(set(mother.genes) | set(father.genes))
    individual = Individual(rng.sample(pool, k=len(mother.genes)))
    return individual


def uniform_mutation(individual: Individual, n_genes: int, prob: float, rng: random.Random) -> Individual:
    """Randomly replace some genes with values not already present in the individual."""
    genes = list(individual.genes)
    for i in range(len(genes)):
        if rng.random() < prob:
            unused = set(range(n_genes)) - set(genes)
            if unused:
                genes[i] = rng.choice(list(unused))
    mutant = Individual(genes)
    return mutant


def tournament_selection(population: Population, rng: random.Random) -> List[Individual]:
    """Select len(population) individuals via pairwise score tournaments."""
    selected = []
    for _ in range(len(population)):
        a, b = rng.sample(population.individuals, 2)
        selected.append(a if a.score > b.score else b)
    return selected
