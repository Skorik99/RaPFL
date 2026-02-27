from ..byzantine_base.byzantine import ByzantineBase

from collections import OrderedDict

import numpy as np
import random
import torch
from hydra.utils import instantiate
from copy import deepcopy

import time


from utils.attack_utils import (
    map_attack_clients,
    set_attack_rounds,
    set_client_map_round,
    load_attack_configs,
    apply_synchronized_attack,
)


class PPBC(ByzantineBase):
    def __init__(self, theta, gamma, **method_args):
        super().__init__()

        self.theta = theta
        self.gamma = gamma

        self.epoch_method = method_args.get("epoch_method", "random")
        self.iter_method = method_args.get("iter_method", "None")
        self.epoch_k = method_args.get("epoch_k", 5)
        self.iter_k = method_args.get("iter_k", 3)
        self.iterations = method_args.get("iterations", 1)

        self.trust_sample_amount = method_args.get("trust_sample_amount", 50)
        self.momentum_beta = method_args.get("momentum_beta", 0.1)
        self.q_m = method_args.get("q_m", 1.0)

    def _init_federated(self, cfg):
        super()._init_federated(cfg)

        self.current_errors_from_clients = {
            f"client {i}": OrderedDict() for i in range(self.num_clients)
        }
        self.final_errors = {
            f"client {i}": OrderedDict() for i in range(self.num_clients)
        }

    def _init_server(self, cfg):
        super()._init_server(cfg)
        # if self.trust_df is not None:
        #    _, self.trust_df = get_stratified_subsample(
        #        df=self.trust_df,
        #        num_samples=self.trust_sample_amount,
        #        random_state=cfg.random_state,
        #    )

        self.num_clients = cfg.federated_params.amount_of_clients
        self.epoch_prev_trust_scores = [1 / self.num_clients] * self.num_clients
        self.iter_prev_trust_scores = [1 / self.num_clients] * self.num_clients

    # =========================================================================#
    #                    Trust Score Calculation Utilities                    #
    # =========================================================================#

    def get_scores_from_gradients(self):
        prev_trust_scores = [0] * self.num_clients
        for rank in range(self.num_clients):
            cur_grads = self.server.client_gradients[rank]
            cur_flat_grad = torch.cat([grad.flatten() for grad in cur_grads.values()])

            prev_trust_scores[rank] = torch.norm(cur_flat_grad)
        return prev_trust_scores

    def get_scores_from_bant(self, server_loss, client_losses):
        uncutted_trust_scores = [
            server_loss - client_loss for client_loss in client_losses
        ]
        trust_scores = [
            max(uncutted_trust_score, 0)
            for uncutted_trust_score in uncutted_trust_scores
        ]

        for i in range(self.num_clients):
            if sum(trust_scores):
                momentum_trust_score = (
                    1 - self.momentum_beta
                ) * self.prev_trust_scores[i] + self.momentum_beta * (
                    trust_scores[i] / sum(trust_scores)
                )
            else:
                momentum_trust_score = (
                    1 - self.momentum_beta
                ) * self.prev_trust_scores[i]
            self.prev_trust_scores[i] = momentum_trust_score

    def get_scores_from_losses(self):
        prev_trust_scores = [0] * self.num_clients
        client_results = self.server.server_metrics
        prev_trust_scores = [metrics[1] for metrics in client_results]

        return prev_trust_scores

    def get_score_from_anglse(self):
        prev_trust_scores = [0] * self.num_clients
        avg_grad = self.get_avg_grad()
        for i in range(len(self.server.client_gradients)):
            client_grad = self.server.client_gradients[i]
            sc_prod = torch.zeros(len(client_grad))
            idx = 0
            for key, value in avg_grad.items():
                sc_prod[idx] = self.get_scalar_prod(value, client_grad[key])
                idx += 1
            prev_trust_scores[i] = torch.mean(sc_prod)
        return prev_trust_scores

    # =========================================================================#
    #               Trust Score Dispatch: Epoch vs. Iteration                 #
    # =========================================================================#

    def _epoch_count_trust_score(self):
        if "bant" in self.epoch_method:
            server_loss, client_losses = self.server.get_trust_losses()
            self.get_scores_from_bant(server_loss, client_losses)

        elif "gradient_norm" in self.epoch_method:
            self.epoch_prev_trust_scores = self.get_scores_from_gradients()

        elif "loss" in self.epoch_method:
            self.epoch_prev_trust_scores = self.get_scores_from_losses()

        elif "angle" in self.epoch_method:
            self.epoch_prev_trust_scores = self.get_score_from_anglse()

        else:
            print(f"{self.epoch_method} method does not requires trust scores")

    def _iter_count_trust_score(self):
        if "bant" in self.iter_method:
            server_loss, client_losses = self.server.get_trust_losses()
            self.get_scores_from_bant(server_loss, client_losses)

        elif "gradient_norm" in self.iter_method:
            self.iter_prev_trust_scores = self.get_scores_from_gradients()

        elif "loss" in self.iter_method:
            self.iter_prev_trust_scores = self.get_scores_from_losses()

        elif "angle" in self.iter_method:
            self.iter_prev_trust_scores = self.get_score_from_anglse()

        else:
            print(f"{self.iter_method} method does not requires trust scores")

    # =========================================================================#
    #                       Some additional functionality                     #
    # =========================================================================#

    def get_avg_grad(self):
        avg_grad = OrderedDict(
            {
                key: torch.zeros_like(value, dtype=torch.float32)
                for key, value in self.server.client_gradients[0].items()
            }
        )
        for i in range(len(self.server.client_gradients)):
            for key, value in self.server.client_gradients[i].items():
                avg_grad[key] += value / float(self.num_clients)

        return avg_grad

    def get_scalar_prod(self, first, second):
        return torch.sum(first * second)

    def get_clients(self):
        bernoulli_dist = torch.distributions.Bernoulli(probs=self.q_m)
        self.probs = bernoulli_dist.sample((self.num_clients,))
        print(f"Availability of clients:\n{self.probs}")

    # =========================================================================#
    #                           Compressor Utilities                          #
    # =========================================================================#
    def random_compressor(self, mode="epoch"):
        if mode == "epoch":
            clients = np.arange(self.num_clients)
            random.shuffle(clients)

            self.epoch_compress_politic = torch.zeros_like(self.current_politic)
            for rank in range(self.epoch_k):
                self.epoch_compress_politic[clients[rank]] = self.current_politic[
                    clients[rank]
                ]

            print(
                clients,
                self.epoch_compress_politic,
                "perm of clients and politic for epoch",
            )

        if mode == "iter":
            clients = torch.arange(self.num_clients)
            nonzero_ranks = list(
                torch.nonzero(self.epoch_compress_politic.cpu(), as_tuple=True)[0]
            )
            random.shuffle(nonzero_ranks)

            self.iter_compress_politic = torch.zeros_like(self.epoch_compress_politic)
            for i in range(self.iter_k):
                self.iter_compress_politic[nonzero_ranks[i]] = (
                    self.epoch_compress_politic[nonzero_ranks[i]]
                )
            print(
                nonzero_ranks,
                self.iter_compress_politic,
                "perm of clients and politic for iter",
            )

    def trust_score_compressor(self, mode="epoch"):
        if mode == "epoch":
            idx_of_k_clients = np.argsort(self.epoch_prev_trust_scores)[::-1][
                : self.epoch_k
            ]

            self.epoch_compress_politic = torch.zeros_like(self.current_politic)
            for rank in range(self.epoch_k):
                self.epoch_compress_politic[idx_of_k_clients[rank]] = (
                    self.current_politic[idx_of_k_clients[rank]]
                )

            print(
                self.epoch_prev_trust_scores,
                self.epoch_compress_politic,
                f"trust scores via {self.epoch_method} of clients and politic for epoch",
            )

        if mode == "iter":
            idx_of_k_clients = np.argsort(self.iter_prev_trust_scores)[::-1]
            nonzero_rank = np.nonzero(self.epoch_compress_politic.cpu())
            best_epoch_results = idx_of_k_clients[
                np.isin(idx_of_k_clients, nonzero_rank)
            ]

            self.iter_compress_politic = torch.zeros_like(self.epoch_compress_politic)
            for rank in range(self.iter_k):
                self.iter_compress_politic[best_epoch_results[rank]] = (
                    self.epoch_compress_politic[best_epoch_results[rank]]
                )

            print(
                self.iter_prev_trust_scores,
                self.iter_compress_politic,
                f"trust scores via {self.iter_method} of clients and politic for iter",
                flush=True,
            )

    def epoch_compressor(self):
        if "random" in self.epoch_method:
            self.random_compressor(mode="epoch")
        else:
            self.trust_score_compressor(mode="epoch")

    def iter_compressor(self):
        if "random" in self.iter_method:
            self.random_compressor(mode="iter")
        else:
            self.trust_score_compressor(mode="iter")

    # =========================================================================#
    #                       Main algorithm functionality                      #
    # =========================================================================#

    def get_init_point(self):
        aggregated_weights = self.server.global_model.state_dict()
        for rank in range(self.num_clients):
            client_errors = self.final_errors[f"client {rank}"]

            for key, _ in aggregated_weights.items():
                aggregated_weights[key] = _ + client_errors[key]

        self.server.global_model.load_state_dict(aggregated_weights)

    def get_errors_on_iter(self, itn):
        aggregated_weights = self.server.global_model.state_dict()

        for rank in range(self.num_clients):
            client_grad = self.server.client_gradients[rank]
            current_client_error = self.current_errors_from_clients[f"client {rank}"]
            current_client_prob = self.probs[rank]
            final_client_error = self.final_errors[f"client {rank}"]
            client_politic = self.iter_compress_politic[rank].to(self.server.device)

            for key, grads in client_grad.items():
                self.current_errors_from_clients[f"client {rank}"][key] = (
                    current_client_error[key]
                    + (1 - self.theta)
                    * (1 / self.num_clients - client_politic)
                    * grads.to(self.server.device)
                    * current_client_prob
                    / self.q_m
                )

                aggregated_weights[key] = (
                    aggregated_weights[key]
                    + self.gamma
                    * (1 - self.theta)
                    * grads.to(self.server.device)
                    * client_politic
                    * current_client_prob
                    / self.q_m
                    + self.gamma * self.theta * final_client_error[key]
                )
            if itn == self.iterations - 1:
                self.final_errors[f"client {rank}"] = self.current_errors_from_clients[
                    f"client {rank}"
                ]

        return aggregated_weights

    def init_errors(self):
        if self.cur_round != 0:
            self.epoch_compressor()
            for rank in range(self.num_clients):
                self.final_errors[f"client {rank}"] = deepcopy(
                    self.current_errors_from_clients[f"client {rank}"]
                )
                for key, _ in self.server.global_model.state_dict().items():
                    self.current_errors_from_clients[f"client {rank}"][key] = (
                        torch.zeros_like(_).to(self.server.device)
                    )
            return
        else:
            self.current_politic = (
                torch.ones(self.num_clients).to(self.server.device) / self.num_clients
            )
            self.epoch_compressor()
            for rank in range(self.num_clients):
                for key, _ in self.server.global_model.state_dict().items():
                    self.current_errors_from_clients[f"client {rank}"][key] = (
                        torch.zeros_like(_).to(self.server.device)
                    )
                    self.final_errors[f"client {rank}"][key] = torch.zeros_like(_).to(
                        self.server.device
                    )
            print("creating the errors is done")

            return

    def process_clients(self):
        self.init_errors()
        self.get_init_point()
        self.get_clients()  # This weights refer to PPBC with unavailible devices (Alg.2)
        for itn in range(self.iterations):
            self.iter_compressor()
            print(f"start the {itn} iteration")
            super().train_round()
            aggregated_weights = self.get_errors_on_iter(itn)

            self._iter_count_trust_score()

            if (itn == self.iterations - 1) and ("bant" in self.epoch_method):
                self._epoch_count_trust_score()

            self.server.global_model.load_state_dict(aggregated_weights)

            print("processing is done")
        if "bant" not in self.epoch_method:
            self._epoch_count_trust_score()

    def check_final_errors(self):
        for i in range(self.num_clients):
            c = 0
            for key, w in self.final_errors[f"client {i}"].items():
                if np.allclose(w.cpu(), torch.zeros_like(w).cpu()):
                    c += 1
            if c == len(self.final_errors[f"client {i}"].items()):
                print(f"all errors for {i} client are equals to zeros")

    def model_l2_norm(self, model):
        all_params = torch.cat([p.detach().flatten() for _, p in model])
        # L2-норма (евклидова)
        return torch.norm(all_params, p=2)

    def begin_train(self):
        self.manager.create_clients(
            self.client_args, self.client_kwargs, self.client_attack_map
        )
        self.server.global_model = instantiate(
            self.cfg.model, num_classes=self.train_dataset.num_classes
        )
        self.list_clients = range(
            0, self.num_clients
        )  # since all clients participate (at least we have to send the model to all clients)
        for round in range(self.rounds):
            self.round = round
            print(f"\nRound number: {round} of {self.rounds}")
            begin_round_time = time.time()
            self.cur_round = round

            self.server.test_global_model()

            print("\nTraining started\n")
            # Setup attack on current round
            self.client_map_round = set_client_map_round(
                self.client_attack_map,
                self.attack_rounds,
                self.attack_scheme,
                round,
            )

            self.process_clients()

            self.server.save_best_model(round)

            self.check_final_errors()

            print(f"Round time: {time.time() - begin_round_time}", flush=True)

        self.manager.stop_train()
