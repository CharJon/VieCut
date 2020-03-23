/******************************************************************************
 * local_search.h
 *
 * Source of VieCut
 *
 ******************************************************************************
 * Copyright (C) 2018-2020 Alexander Noe <alexander.noe@univie.ac.at>
 *
 * Published under the MIT license in the LICENSE file.
 *****************************************************************************/

#pragma once

#include <memory>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/configuration.h"
#include "common/definitions.h"

class local_search {
 private:
    std::shared_ptr<multicut_problem> problem;
    const mutable_graph& original_graph;
    const std::vector<NodeID>& original_terminals;
    const std::vector<bool>& fixed_vertex;
    std::vector<NodeID>* sol;
    std::unordered_map<NodeID, NodeID> movedToNewBlock;
    std::vector<std::vector<FlowType> > previousConnectivity;
    std::vector<NodeID> noImprovement;

 public:
    local_search(std::shared_ptr<multicut_problem> problem,
                 const mutable_graph& original_graph,
                 const std::vector<NodeID>& original_terminals,
                 const std::vector<bool>& fixed_vertex,
                 std::vector<NodeID>* sol)
        : problem(problem),
          original_graph(original_graph),
          original_terminals(original_terminals),
          fixed_vertex(fixed_vertex),
          sol(sol),
          previousConnectivity(original_terminals.size()) {
        for (auto& pc : previousConnectivity) {
            pc.resize(original_terminals.size(), 0);
        }
    }

 private:
    std::tuple<EdgeWeight, FlowType>
    flowBetweenBlocks(NodeID term1, NodeID term2, bool fixEdges) {
        std::vector<NodeID>& solution = *sol;
        std::vector<NodeID> mapping(original_graph.n(), UNDEFINED_NODE);
        auto G = std::make_shared<mutable_graph>();
        FlowType sol_weight = 0;
        timer t;

        NodeID id = 2;
        for (NodeID n : original_graph.nodes()) {
            if (solution[n] != term1 && solution[n] != term2)
                continue;
            if (fixed_vertex[n]) {
                mapping[n] = (solution[n] == term1 ? 0 : 1);
            } else {
                mapping[n] = id;
                id++;
            }
        }
        G->start_construction(id);
        std::unordered_map<NodeID, EdgeWeight> edgesToFixed0;
        std::unordered_map<NodeID, EdgeWeight> edgesToFixed1;
        std::vector<std::pair<NodeID, NodeID> > edgesOnOriginal;

        for (NodeID n : original_graph.nodes()) {
            if (solution[n] != term1 && solution[n] != term2)
                continue;
            NodeID m_n = mapping[n];
            for (EdgeID e : original_graph.edges_of(n)) {
                auto [t, w] = original_graph.getEdge(n, e);
                NodeID m_t = mapping[t];
                if ((solution[t] != term1 && solution[t] != term2)
                    || m_n >= m_t || m_t < 2)
                    continue;

                if (solution[t] != solution[n]) {
                    sol_weight += w;
                    edgesOnOriginal.emplace_back(n, t);
                }

                if (m_n < 2) {
                    if (m_n == 0) {
                        if (edgesToFixed0.count(m_t) > 0) {
                            edgesToFixed0[m_t] += w;
                        } else {
                            edgesToFixed0[m_t] = w;
                        }
                    } else {
                        if (edgesToFixed1.count(m_t) > 0) {
                            edgesToFixed1[m_t] += w;
                        } else {
                            edgesToFixed1[m_t] = w;
                        }
                    }
                } else {
                    G->new_edge_order(m_n, m_t, w);
                }
            }
        }
        for (auto [n, w] : edgesToFixed0) {
            G->new_edge_order(n, 0, w);
        }
        for (auto [n, w] : edgesToFixed1) {
            G->new_edge_order(n, 1, w);
        }

        std::vector<NodeID> terminals = { 0, 1 };
        push_relabel pr;
        LOG0 << "build " << t.elapsed();
        auto [f, s] = pr.solve_max_flow_min_cut(G, terminals, 0, true);
        std::unordered_set<NodeID> zero;
        sLOG0 << "flow " << t.elapsed() << term1 << term2;
        for (NodeID v : s) {
            zero.insert(v);
        }

        if (f < sol_weight) {
            LOG0 << term1 << "-" << term2 << ": "
                 << sol_weight << " to " << f;
        } else {
            noImprovement.emplace_back(f);
        }
        size_t improvement = (sol_weight - f);
        std::vector<NodeID> movedVertices;
        for (NodeID n : original_graph.nodes()) {
            if (solution[n] == term1 || solution[n] == term2) {
                if (fixed_vertex[n]) {
                    if (zero.count(mapping[n]) != (solution[n] == term1)) {
                        LOG1 << "DIFFERENT";
                        exit(1);
                    }
                }

                NodeID map = mapping[n];

                if (zero.count(map) > 0) {
                    solution[n] = term1;
                } else {
                    solution[n] = term2;
                }
            }
        }

        if ((f <= 5 || fixEdges) && configuration::getConfig()->inexact) {
            // inexact: let's just say these are optimal and delete the edges
            for (auto [a, b] : edgesOnOriginal) {
                LOG0 << "edge " << a << " " << b;
                NodeID pa = problem->graph->getCurrentPosition(a);
                NodeID pb = problem->graph->getCurrentPosition(b);

                NodeID pia = problem->graph->getPartitionIndex(pa);
                NodeID pib = problem->graph->getPartitionIndex(pb);
                if (pa >= problem->graph->n() || pb >= problem->graph->n()) {
                    LOG1 << "edge too large";
                    continue;
                }
                if ((pia == term1 || pia == term2) &&
                    (pib == term1 || pib == term2) && pia != pib) {
                    for (EdgeID e : problem->graph->edges_of(pa)) {
                        auto [t, w] = problem->graph->getEdge(pa, e);
                        if (t == pb) {
                            problem->graph->deleteEdge(pa, e);
                            problem->deleted_weight += w;
                        }
                    }
                }
            }
            problem->addFinishedPair(term1, term2, original_terminals.size());
        }
        LOG0 << "done " << t.elapsed();

        return std::make_tuple(improvement, f);
    }

    void fixLargestFlow() {
        std::vector<NodeID>& solution = *sol;
        size_t numflows = original_terminals.size();
        LOG1 << numflows << " numflows";
        std::vector<std::vector<FlowType> > blockConnectivity(numflows);
        for (auto& b : blockConnectivity) {
            b.resize(numflows, 0);
        }

        for (NodeID n : original_graph.nodes()) {
            NodeID blockn = solution[n];
            for (EdgeID e : original_graph.edges_of(n)) {
                auto [t, w] = original_graph.getEdge(n, e);
                if (solution[t] > blockn) {
                    if (!fixed_vertex[n] || !fixed_vertex[t]) {
                        blockConnectivity[blockn][solution[t]] += w;
                    }
                }
            }
        }

        NodeID maxi = 0;
        NodeID maxj = 0;
        FlowType maxcon = 0;
        for (size_t i = 0; i < blockConnectivity.size(); ++i) {
            for (size_t j = 0; j < blockConnectivity[i].size(); ++j) {
                FlowType connect = blockConnectivity[i][j];
                if (connect > maxcon) {
                    maxi = i;
                    maxj = j;
                    maxcon = connect;
                }
            }
        }
        LOG1 << "max connect from " << maxi << " to " << maxj << ": " << maxcon;
        flowBetweenBlocks(maxi, maxj, true);
        LOG1 << "flowed";
        problem->addFinishedPair(maxi, maxj, original_terminals.size());
        LOG1 << "addingt";
    }

    EdgeWeight flowLocalSearch() {
        std::vector<NodeID>& solution = *sol;
        std::vector<std::vector<FlowType> >
        blockConnectivity(original_terminals.size());

        EdgeWeight improvement = 0;

        std::vector<std::tuple<NodeID, NodeID, FlowType> > neighboringBlocks;

        for (auto& b : blockConnectivity) {
            b.resize(original_terminals.size(), 0);
        }

        for (NodeID n : original_graph.nodes()) {
            NodeID blockn = solution[n];
            for (EdgeID e : original_graph.edges_of(n)) {
                auto [t, w] = original_graph.getEdge(n, e);
                if (solution[t] > blockn) {
                    if (!fixed_vertex[n] || !fixed_vertex[t]) {
                        blockConnectivity[blockn][solution[t]] += w;
                    }
                }
            }
        }

        for (size_t i = 0; i < blockConnectivity.size(); ++i) {
            for (size_t j = 0; j < blockConnectivity[i].size(); ++j) {
                FlowType connect = blockConnectivity[i][j];
                if (connect != previousConnectivity[i][j]) {
                    neighboringBlocks.emplace_back(i, j, connect);
                }
            }
        }

        random_functions::permutate_vector_good(&neighboringBlocks);
        //    std::sort(neighboringBlocks.begin(), neighboringBlocks.end(),
        //        [](const auto& n1, const auto& n2) {
        //            return std::get<2>(n1) > std::get<2>(n2);
        //        });

        EdgeWeight minconnect = 500000;
        for (auto [a, b, c] : neighboringBlocks) {
            if (!problem->isPairFinished(a, b, original_terminals.size())) {
                auto [impr, connect] = flowBetweenBlocks(a, b, false);
                improvement += impr;
                previousConnectivity[a][b] = connect;
                if (impr > 0 && connect < minconnect) {
                    minconnect = connect;
                }
            }
        }
        noImprovement.clear();

        return improvement;
    }

    EdgeWeight gainLocalSearch() {
        bool inexact = configuration::getConfig()->inexact;
        FlowType improvement = 0;
        std::vector<NodeID>& current_solution = *sol;
        std::vector<NodeID> permute(original_graph.n(), 0);
        std::vector<bool> inBoundary(original_graph.n(), true);
        std::vector<std::pair<NodeID, int64_t> > nextBest(
            original_graph.n(), { UNDEFINED_NODE, 0 });

        std::vector<bool> isTerm(problem->graph->n(), false);
        for (auto t : problem->terminals) {
            isTerm[t.position] = true;
        }

        random_functions::permutate_vector_good(&permute, true);

        for (NodeID v : original_graph.nodes()) {
            NodeID n = permute[v];
            NodeID o = problem->mapped(n);
            NodeID pos = problem->graph->getCurrentPosition(o);
            if (fixed_vertex[n] || !inBoundary[n] || isTerm[pos])
                continue;

            std::vector<EdgeWeight> blockwgt(
                configuration::getConfig()->num_terminals, 0);
            NodeID ownBlockID = current_solution[n];
            for (EdgeID e : original_graph.edges_of(n)) {
                auto [t, w] = original_graph.getEdge(n, e);
                NodeID block = current_solution[t];
                blockwgt[block] += w;
            }

            EdgeWeight ownBlockWgt = blockwgt[ownBlockID];
            NodeID maxBlockID = 0;
            EdgeWeight maxBlockWgt = 0;
            for (size_t i = 0; i < blockwgt.size(); ++i) {
                if (i != ownBlockID) {
                    if (blockwgt[i] > maxBlockWgt) {
                        maxBlockID = i;
                        maxBlockWgt = blockwgt[i];
                    }
                }
            }

            if (maxBlockWgt) {
                inBoundary[n] = false;
            }

            int64_t gain = static_cast<int64_t>(maxBlockWgt)
                           - static_cast<int64_t>(ownBlockWgt);

            bool notDoublemoved = true;
            for (EdgeID e : original_graph.edges_of(n)) {
                auto [t, w] = original_graph.getEdge(n, e);
                auto [nbrBlockID, nbrGain] = nextBest[t];
                int64_t movegain = nbrGain + gain + 2 * w;
                if (current_solution[t] == current_solution[n] &&
                    nbrBlockID == maxBlockID && movegain > 0
                    && movegain > gain) {
                    current_solution[n] = maxBlockID;
                    current_solution[t] = maxBlockID;
                    improvement += movegain;
                    if (inexact) {
                        movedToNewBlock[n] = maxBlockID;
                        movedToNewBlock[t] = maxBlockID;
                    }

                    notDoublemoved = false;

                    for (EdgeID e : original_graph.edges_of(n)) {
                        NodeID b = original_graph.getEdgeTarget(n, e);
                        nextBest[b] = std::make_pair(UNDEFINED_NODE, 0);
                        inBoundary[b] = true;
                    }
                    nextBest[t] = std::make_pair(UNDEFINED_NODE, 0);
                    for (EdgeID e : original_graph.edges_of(t)) {
                        NodeID b = original_graph.getEdgeTarget(t, e);
                        nextBest[b] = std::make_pair(UNDEFINED_NODE, 0);
                        inBoundary[b] = true;
                    }
                }
            }

            if (!notDoublemoved) {
                continue;
            }

            if (gain >= 0) {
                current_solution[n] = maxBlockID;
                if (inexact) {
                    movedToNewBlock[n] = maxBlockID;
                }
                improvement += gain;
                for (EdgeID e : original_graph.edges_of(n)) {
                    NodeID t = original_graph.getEdgeTarget(n, e);
                    nextBest[t] = std::make_pair(UNDEFINED_NODE, 0);
                    inBoundary[t] = true;
                }
            } else {
                nextBest[n] = std::make_pair(maxBlockID, gain);
            }
        }
        return improvement;
    }

 public:
    FlowType improveSolution() {
        FlowType total_improvement = 0;
        bool change_found = true;
        size_t ls_iter = 0;
        while (change_found) {
            timer t;
            change_found = false;
            auto impGain = gainLocalSearch();
            total_improvement += impGain;
            LOG1 << "gain " << t.elapsed();
            auto impFlow = flowLocalSearch();
            total_improvement += impFlow;

            if (impFlow > 0 || impGain > 0)
                change_found = true;

            LOG1 << "local search iteration " << ls_iter++ << " complete - t:"
                 << t.elapsed() << " flow:" << impFlow << " gain:" << impGain;
        }
        // fixLargestFlow();
        return total_improvement;
    }

    void contractMovedVertices() {
        std::vector<std::unordered_set<NodeID> > ctrSets(
            original_terminals.size());
        std::vector<bool> isTerm(problem->graph->n(), false);

        for (auto t : problem->terminals) {
            ctrSets[t.original_id].insert(t.position);
            isTerm[t.position] = true;
        }

        for (size_t i = 0; i < ctrSets.size(); ++i) {
            for (auto [v, nb] : movedToNewBlock) {
                if (nb == i) {
                    NodeID m = problem->mapped(v);
                    NodeID curr = problem->graph->getCurrentPosition(m);
                    if (!isTerm[curr]) {
                        ctrSets[nb].insert(curr);
                    }
                }
            }
            if (ctrSets[i].size() > 1) {
                problem->graph->contractVertexSet(ctrSets[i]);
            }
            graph_contraction::setTerminals(problem, original_terminals);
        }
        graph_contraction::deleteTermEdges(problem, original_terminals);
    }
};
