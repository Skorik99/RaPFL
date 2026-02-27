from ..ppbc.ppbc import PPBC
from .rap_server import RapServer
from hydra.utils import instantiate
import torch
from copy import deepcopy
import numpy as np


class RaP(PPBC):
    def __init__(
        self,
        theta,
        gamma,
        start_point,
        end_point,
        num_opt_epochs,
        mirror_gamma,
        ts_momentum,
        proba,
        bant_epochs,
        ppbc_moment,
        surrogate_sending,
        autobant_epochs,
        **method_args,
    ):
        super().__init__(theta, gamma, **method_args)
        self.opt_params = [
            start_point,
            end_point,
            num_opt_epochs,
            mirror_gamma,
            ts_momentum,
        ]
        self.iter_proba = proba
        self.bant_epochs = bant_epochs
        self.ppbc_moment = ppbc_moment
        self.surrogate_sending = surrogate_sending
        self.autobant_epochs = autobant_epochs

    def _init_server(self, cfg):
        self.trust_df = None
        if cfg.get("trust_dataset"):
            self.trust_df = instantiate(
                cfg.trust_dataset, cfg=cfg, mode="trust", _recursive_=False
            )
        self.server = RapServer(cfg, self.trust_df, *self.opt_params)
        self.num_clients = cfg.federated_params.amount_of_clients
        self.epoch_prev_trust_scores = [1 / self.num_clients] * self.num_clients
        self.iter_k = self.num_clients_subset
        self.epoch_k = self.amount_of_clients
        self.server.gamma = self.gamma
        self.server.theta = self.theta

    def get_init_point(self):
        pass  # since x^{k-1, H^{k-1}} == x^{k,0}

    def get_errors_on_iter(self, itn):
        aggregated_weights = self.server.global_model.state_dict()

        if self.surrogate_sending == "jointly":
            if itn == 0 and self.round == 0:
                self.server.final_errors = self.final_errors
            else:
                for client in self.chosen_clients:
                    self.server.final_errors[f"client {client}"] = deepcopy(
                        self.current_errors_from_clients[f"client {client}"]
                    )
        else:
            if itn == 0:
                self.server.final_errors = self.final_errors

        self.server.list_clients = self.chosen_clients
        self.server._init_criterion()
        self.server._init_trust_model()

        if self.round < self.autobant_epochs:
            print("Now we count grad_weights")
            grad_weights = self.server._count_trust_score_manager("grad")

            print("Now we count surrogate_weights")
            surrogate_weights = (
                self.server._count_trust_score_manager("surrogate")
                if self.round != 0
                else [1 / self.amount_of_clients] * self.amount_of_clients
            )
            if self.round > 0:
                surrogate_weights = self.server._check_w(surrogate_weights)
        else:
            surrogate_weights = self.server.best_surrogate
            grad_weights = torch.tensor([0.0] * len(self.chosen_clients))
            for i in range(len(self.chosen_clients)):
                client = self.chosen_clients[i]
                grad_weights[i] = surrogate_weights[client]

        print("Start validate grad weights with surrogate weights:")
        for i in range(len(grad_weights)):
            client = self.chosen_clients[i]
            if surrogate_weights[client] == 0:
                grad_weights[i] = 0
                print(
                    f"change in client {self.chosen_clients[i]}:"
                    f"grad={grad_weights[i]} surrogate={surrogate_weights[client]}"
                )
        print("End validate")

        if abs(sum(grad_weights) - 1) > 1e-5 and sum(grad_weights) != 0:
            print(f"We renorm grad_weight, sum = {sum(grad_weights)}")
            grad_weights = grad_weights / sum(grad_weights)

        if (abs(sum(surrogate_weights) - 1) > 1e-5) and (self.cur_round != 0):
            print(f"We renorm surrogate_weights, sum = {sum(surrogate_weights)}")
            surrogate_weights = surrogate_weights / sum(surrogate_weights)

        print("\n\n\nCalculated grad weights:")
        for i in range(len(grad_weights)):
            client = self.chosen_clients[i]
            print(f"for client {client}: {grad_weights[i]}")

        print("\n\n\nCalcualated surrogate weights:")
        for i in range(self.amount_of_clients):
            print(f"for client {i}: {surrogate_weights[i]}")

        print("\n\n\n")

        # Add aggregated weights with surrogate gradients (we need all clients)
        for rank in range(self.num_clients):
            client_grad = self.server.client_gradients[rank]
            current_client_error = self.current_errors_from_clients[f"client {rank}"]
            current_client_prob = self.probs[rank]
            final_client_error = self.final_errors[f"client {rank}"]
            surrogate_weight = surrogate_weights[rank]

            for key, grads in client_grad.items():
                self.current_errors_from_clients[f"client {rank}"][key] = (
                    current_client_error[key] * self.ppbc_moment
                    + (1 - self.theta)
                    * (1 - self.ppbc_moment)
                    * grads.to(self.server.device)
                    * current_client_prob
                    / self.q_m
                )

                aggregated_weights[key] = (
                    aggregated_weights[key]
                    + self.gamma
                    * self.theta
                    * final_client_error[key]
                    * surrogate_weight
                )

        # add aggregated wrights with gradients of this iteration (we need only self.chosen_clients)
        for it in range(len(self.chosen_clients)):
            rank = self.chosen_clients[it]
            grad_weight = grad_weights[it]
            client_grad = self.server.client_gradients[rank]
            for key, grads in client_grad.items():
                aggregated_weights[key] = (
                    aggregated_weights[key]
                    + self.gamma
                    * (1 - self.theta)
                    * grads.to(self.server.device)
                    * grad_weight
                )
        return aggregated_weights

    def init_errors(self):
        if self.cur_round != 0:
            for rank in range(self.num_clients):
                self.final_errors[f"client {rank}"] = deepcopy(
                    self.current_errors_from_clients[f"client {rank}"]
                )
        else:
            for rank in range(self.num_clients):
                for key, _ in self.server.global_model.state_dict().items():
                    self.current_errors_from_clients[f"client {rank}"][key] = (
                        torch.zeros_like(_).to(self.server.device)
                    )
                    self.final_errors[f"client {rank}"][key] = torch.zeros_like(_).to(
                        self.server.device
                    )
        print("creating the errors is done")

    def check_final_errors(self):
        pass

    def process_clients(self):
        self.iterations = np.random.geometric(p=self.iter_proba)
        print(f"Number of iterations for this round: {self.iterations}")
        self.init_errors()
        self.get_init_point()
        self.server.cur_round = self.round

        for itn in range(self.iterations):
            print(f"start the {itn} iteration")
            self.get_clients()  # This weights refer to PPBC with unavailible devices (Alg.2)
            self.chosen_clients = self.server.select_clients_to_train(
                self.num_clients_subset
            )
            print(f"Chosen clients: {self.chosen_clients}")
            self.train_round()
            aggregated_weights = self.get_errors_on_iter(itn)

            self.server.global_model.load_state_dict(aggregated_weights)

            if self.iterations > 10 and itn == self.iterations // 2:
                self.server.test_global_model()

            print("processing is done")
