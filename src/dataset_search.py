from itertools import combinations_with_replacement

from utils.coloring.graph_color import color_generator
from utils.generator.graph_generator import graph_generator
from utils.tagger.label_generator import Tagger


def stats(number_graphs,
          graphs,
          tagger):

    total_nodes = 0
    total_1s = 0
    total_graph_1s = 0
    all_1s = 0
    all_0s = 0
    avg_1s_not_all_1s = 0
    not_all_1s_size = 0

    for graph in graphs:

        graph, num_nodes, num_ones, graph_label = tagger(graph=graph)

        total_nodes += num_nodes
        total_1s += num_ones
        total_graph_1s += graph_label

        _all_1s = num_ones == len(graph)
        _all_0s = num_ones == 0

        all_1s += int(_all_1s)
        all_0s += int(_all_0s)
        avg_1s_not_all_1s += num_ones if (
            not _all_1s and not _all_0s) else 0

        not_all_1s_size += num_nodes if (
            not _all_1s and not _all_0s) else 0

    total_1s_ = float(total_1s) / total_nodes
    all_1s_ = float(all_1s) / number_graphs
    all_0s_ = float(all_0s) / number_graphs
    with_0s_1s_ = float(number_graphs - all_0s - all_1s) / number_graphs

    if (number_graphs - all_0s - all_1s) > 0:
        # average number of ones per graph in graph with not all 1s
        avg_1s_not_all_1s = float(avg_1s_not_all_1s) / \
            (number_graphs - all_0s - all_1s)
        # average size of graphs with not all 1s
        not_all_1s_size = float(not_all_1s_size) / \
            (number_graphs - all_0s - all_1s)

        avg_1s_not_all_1s_ = float(avg_1s_not_all_1s) / not_all_1s_size

    else:
        avg_1s_not_all_1s_ = 0

    ######################

    if total_1s_ > 0.9:
        return False, f"Total 1s greater than 0.9: {total_1s_}", (
            total_1s_, all_1s_, all_0s_, with_0s_1s_, avg_1s_not_all_1s_)
    if all_0s_ > 0.4:
        return False, f"All 0s greater than 0.4: {all_0s_}", (
            total_1s_, all_1s_, all_0s_, with_0s_1s_, avg_1s_not_all_1s_)
    if all_1s_ > 0.4:
        return False, f"All 1s greater than 0.4: {all_1s_}", (
            total_1s_, all_1s_, all_0s_, with_0s_1s_, avg_1s_not_all_1s_)
    if with_0s_1s_ < 0.3:
        return False, f"0s and 1s less than 0.3: {with_0s_1s_}", (
            total_1s_, all_1s_, all_0s_, with_0s_1s_, avg_1s_not_all_1s_)

    return True, "", (total_1s_, all_1s_, all_0s_,
                      with_0s_1s_, avg_1s_not_all_1s_)


def generate_dataset(number_graphs,
                     generator_fn,
                     n_nodes,
                     structure_fn,
                     formula,
                     seed=None,
                     number_colors=10,
                     graph_split=None,
                     color_distributions=None,
                     **kwargs):

    min_nodes, max_nodes = n_nodes

    tagger = Tagger(formula, **kwargs)
    generator = graph_generator(generator_fn=generator_fn,
                                min_nodes=min_nodes,
                                max_nodes=max_nodes,
                                **kwargs)
    color_graphs = color_generator(graph_generator=generator,
                                   number_graphs=number_graphs,
                                   min_nodes=min_nodes,
                                   max_nodes=max_nodes,
                                   graph_split=graph_split,
                                   color_distributions=color_distributions,
                                   structure_fn=structure_fn,
                                   n_colors=number_colors,
                                   **kwargs)

    return stats(number_graphs=number_graphs,
                 graphs=color_graphs,
                 tagger=tagger)


if __name__ == "__main__":

    m = 2

    fail_filename = f"logging/search/fail_combinations_m{m}.log"
    work_filename = f"logging/search/work_combinations_m{m}.log"

    with open(fail_filename, "w") as fail_file, \
            open(work_filename, "w") as work_file:

        fail_file.write(
            "m,min_green,max_green,p1,p2,p3,total_1s,all_1s,all_0s,with_0s_1s,avg_1s_not_all_1s,reason\n")
        work_file.write(
            "m,min_green,max_green,p1,p2,p3,total_1s,all_1s,all_0s,with_0s_1s,avg_1s_not_all_1s\n")

        for min_green, max_green in \
                combinations_with_replacement(range(10, 20), 2):
            for p1 in range(12, 16):
                for p2 in range(15, 36):
                    for p3 in range(15, 36):

                        if (p2, p3) in combinations_with_replacement(
                                range(25, 36), 2):
                            continue

                        print(
                            f"m:{m} minG:{min_green} maxG:{max_green} p1:{p1} p2:{p2} p3:{p3}")

                        works, msg, \
                            (t_1s, a_1s, a_0s, w_0s_1s, avg_1s_w0s1s) = \
                            generate_dataset(number_graphs=500,
                                             # empty|degree|line|random|cycle
                                             generator_fn="random",
                                             n_nodes=(50, 50),
                                             # line|cycle|normal|centroid
                                             structure_fn="normal",
                                             # formula{1|2|3}
                                             formula="formula4",
                                             seed=None,
                                             number_colors=5,
                                             # global, tuple
                                             greens=(min_green, max_green),
                                             # random
                                             name="erdos",
                                             m=m,
                                             # formula 3
                                             local_prop=[],
                                             global_prop=[0],
                                             global_constraint={0: p1},
                                             condition="and",
                                             # formula 4
                                             nested=[
                                                 "formula4",
                                                 "formula3"
                                             ],
                                             local_prop_nested=[
                                                 [],
                                                 []
                                             ],
                                             constraint_nested=[
                                                 p3,
                                                 p2
                                             ],
                                             self_satisfy_nested=[
                                                 False,
                                                 False
                                             ])

                        if works:
                            work_file.write(
                                f"{m},{min_green},{max_green},{p1},{p2},{p3},{t_1s},{a_1s},{a_0s},{w_0s_1s},{avg_1s_w0s1s}\n")
                        else:
                            fail_file.write(
                                f"{m},{min_green},{max_green},{p1},{p2},{p3},{t_1s},{a_1s},{a_0s},{w_0s_1s},{avg_1s_w0s1s},\"{msg}\"\n")
