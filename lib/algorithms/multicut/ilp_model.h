/******************************************************************************
 * ilp_model.h
 *
 * Source of VieCut
 *
 ******************************************************************************
 * Copyright (C) 2017 Alexandra Henzinger <ahenz@stanford.edu>
 * Copyright (C) 2017-2019 Alexander Noe <alexander.noe@univie.ac.at>
 *
 * Published under the MIT license in the LICENSE file.
 *****************************************************************************/

#pragma once

#include <cmath>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "gurobi_c++.h" // NOLINT

#include "algorithms/multicut/multicut_problem.h"
#include "common/configuration.h"
#include "tools/timer.h"

class reducedGraph {
public:
    int n;
    int m;
    std::vector<int> nodes;

    std::vector<std::tuple<NodeID, NodeID, double, int>> edges;

    reducedGraph(int num_nodes, int num_edges) : n(num_nodes), m(num_edges)
    {
        nodes = std::vector<int>(n, -1);
    }

    void addEdge(NodeID u, NodeID v, double weight, int edgeAssignment) {
        edges.emplace_back(u, v, weight, edgeAssignment);
    }

    void addNode(NodeID u, int nodeAssignment) {
        nodes[u] = nodeAssignment;
    }

    void writeGraph(std::string filename) {
        std::ofstream out(filename);
        out << n << " " << m << std::endl;

        for (int i = 0; i < n; ++i) {
            out << i << " " << nodes[i] << std::endl;
        }

        for (auto& [u, v, weight, edgeAssignment] : edges) {
            out << u << " " << v << " " << weight << " " << edgeAssignment << std::endl;
        }
        out.close();
    }
};

class ilp_model {
 public:
    std::tuple<std::vector<NodeID>, EdgeWeight, bool> computeIlp(
        problemPointer problem,
        const std::vector<NodeID>& presets,
        size_t num_terminals,
        bool parallel,
        size_t thread_id) {
        //multicut_problem::writeGraph(problem, "ilp_graph");


        try {

            mutableGraphPtr graph = problem->graph;
            timer ilp_timer;
            LOG1 << "starting ilp on graph with " << graph->n() << " vertices "
                 << "and " << graph->m() << " edges!";
            GRBModel model = GRBModel(ilp_model::env);

            reducedGraph ilp_graph(graph->n(), graph->m() / 2);

            NodeID max_weight = 0;
            NodeID max_id = 0;
            for (size_t i = 0; i < problem->terminals.size(); ++i) {
                NodeID v = problem->terminals[i].position;
                if (graph->getWeightedNodeDegree(v) > max_weight) {
                    max_weight = graph->getWeightedNodeDegree(v);
                    max_id = i;
                }
            }

            std::vector<std::vector<GRBVar> > nodes(num_terminals);
            std::vector<GRBVar> edges(graph->m() / 2);

            for (auto& v : nodes) {
                v.resize(graph->n());
            }

            model.set(GRB_StringAttr_ModelName, "Partition");
            model.set(GRB_DoubleParam_MIPGap, 0);
            model.set(GRB_IntParam_LogToConsole, 0);
            if (!parallel) {
                model.set(GRB_IntParam_Threads, 1);
            } else {
                size_t threads = configuration::getConfig()->threads;
                model.set(GRB_IntParam_Threads, threads);
                LOG1 << "Running parallel ILP with "
                     << model.get(GRB_IntParam_Threads) << " threads";
                if (!configuration::getConfig()->disable_cpu_affinity) {
                    cpu_set_t all_cores;
                    CPU_ZERO(&all_cores);
                    for (size_t i = 0; i < threads; ++i) {
                        CPU_SET(i, &all_cores);
                    }

                    sched_setaffinity(0, sizeof(cpu_set_t), &all_cores);
                }
            }
            model.set(GRB_IntParam_PoolSearchMode, 0);
            model.set(GRB_DoubleParam_TimeLimit,
                      configuration::getConfig()->ilpTime);
            // Set decision variables for nodes
            for (size_t q = 0; q < num_terminals; q++) { // loop over terminals
                GRBLinExpr nodeTot = 0;
                for (NodeID i = 0; i < graph->n(); i++) { // loop over nodes
                    if (presets[i] < num_terminals) { // if node is assigned to any terminal
                        bool isCurrent = (presets[i] == q); // if node is already assigned to this terminal
                        double f = isCurrent ? 1.0 : 0.0;
                        nodes[q][i] = model.addVar(f, f, 0, GRB_BINARY); // then fix this value to 1, else 0
                        nodes[q][i].set(GRB_DoubleAttr_Start, f);
                    } else {
                        nodes[q][i] = model.addVar(0.0, 1.0, 0, GRB_BINARY);
                        nodes[q][i].set(GRB_DoubleAttr_Start, (max_id == q));
                    }
                }
            }

            for (NodeID i = 0; i < graph->n(); i++) { // loop over nodes
                if (i < num_terminals) {
                    ilp_graph.addNode(i, -2); // is terminal itself
                }
                else if (presets[i] < num_terminals) {
                    ilp_graph.addNode(i, presets[i]); // assigned to a terminal
                }
                else
                {
                    ilp_graph.addNode(i, -1); // not assigned to any terminal
                }
            }

            size_t j = 0;
            // Decision variables for edges
            GRBLinExpr edgeTot = 0;
            for (NodeID n : graph->nodes()) {
                for (EdgeID e : graph->edges_of(n)) {
                    auto [t, w] = graph->getEdge(n, e);
                    if (n > t) {
                        bool terminalIncident = (presets[n] < num_terminals ||
                                                 presets[t] < num_terminals);

                        edges[j] = model.addVar(0.0, 1.0, w, GRB_BINARY);
                        // there is no edge between terminals, we mark edges
                        // that are incident to non-maximal weight terminal
                        double start =
                            (terminalIncident && presets[n] != max_id
                             && presets[t] != max_id) ? 1.0 : 0.0;

                        edges[j].set(GRB_DoubleAttr_Start, start);

                        ilp_graph.addEdge(n, t, w, start);

                        for (size_t q = 0; q < num_terminals; q++) {
                            GRBLinExpr c = nodes[q][n] - nodes[q][t];
                            // Add constraint: valid partiton
                            std::string v =
                                "valid part on edge " + std::to_string(j)
                                + " between " + std::to_string(n)
                                + " and " + std::to_string(t);
                            std::string w =
                                "neg valid part on edge "
                                + std::to_string(j) + " between "
                                + std::to_string(n) + " and "
                                + std::to_string(t);
                            model.addConstr(edges[j], GRB_GREATER_EQUAL, c, v);
                            model.addConstr(edges[j], GRB_GREATER_EQUAL, -c, w);
                        }
                        j++;
                    }
                }
            }


            auto config = configuration::getConfig();

            if (config->reduced_path != "")
            {
                ilp_graph.writeGraph(config->reduced_path + "_" + std::to_string(config->num_reduced_graphs) + ".mtc");
                config->num_reduced_graphs++;
            }


            // Add constraint: sum of all decision variables for 1 node is 1
            for (size_t i = 0; i < graph->n(); i++) {
                GRBLinExpr sumCons = 0;
                for (size_t q = 0; q < num_terminals; q++) {
                    sumCons += nodes[q][i];
                }
                model.addConstr(sumCons, GRB_EQUAL, 1);
            }

            model.set(GRB_IntAttr_ModelSense, GRB_MINIMIZE);
            // Optimize model
            model.optimize();

            std::vector<NodeID> result(graph->n());
            // if solution is found
            if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL ||
                model.get(GRB_IntAttr_Status) == GRB_TIME_LIMIT) {
                // set partition
                for (PartitionID q = 0; q < num_terminals; q++) {
                    for (size_t i = 0; i < graph->n(); i++) {
                        auto v = nodes[q][i].get(GRB_DoubleAttr_X);
                        if (v == 1) {
                            result[i] = q;
                        }
                    }
                }
            } else {
                LOG1 << "No solution";
            }

            bool reIntroduce = false;
            if (model.get(GRB_IntAttr_Status) == GRB_TIME_LIMIT) {
                LOG1 << "ILP Timeout - Re-introducing problem to queue!";
                reIntroduce = true;
            }
            EdgeWeight wgt = std::lround(model.get(GRB_DoubleAttr_ObjVal));

            if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL ||
                model.get(GRB_IntAttr_Status) == GRB_TIME_LIMIT) {
                bool optimal = (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL);

                LOG1 << "SOLUTION time=" << ilp_timer.elapsed()
                     << " graph=" << configuration::getConfig()->graph_filename
                     << " n=" << graph->n()
                     << " m=" << graph->m()
                     << " wgt=" << wgt
                     << " total=" << wgt + problem->deleted_weight
                     << " original_terminals=" << num_terminals
                     << " current_terminals=" << problem->terminals.size()
                     << " optimal_finish=" << optimal;
            }

            if (parallel) {
                if (!configuration::getConfig()->disable_cpu_affinity) {
                    cpu_set_t my_id;
                    CPU_ZERO(&my_id);
                    CPU_SET(thread_id, &my_id);
                    sched_setaffinity(0, sizeof(cpu_set_t), &my_id);
                }
            }

            return std::make_tuple(result, wgt, reIntroduce);
        } catch (GRBException e) {
            LOG1 << e.getErrorCode() << " Message: " << e.getMessage();
            exit(1);
        }
    }

 private:
    GRBEnv env;
};
