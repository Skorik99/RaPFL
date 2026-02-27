from collections import OrderedDict
import copy
from ..autobant.autobant_server import AutoBANTServer
from ..autobant.autobant_models import AutoBANTModel2d
import numpy as np
import torch.nn as nn
import torch


class RapServer(AutoBANTServer):
    def __init__(
        self,
        cfg,
        trust_df,
        start_point,
        end_point,
        num_opt_epochs,
        mirror_gamma,
        ts_momentum,
    ):
        super().__init__(
            cfg,
            trust_df,
            start_point,
            end_point,
            num_opt_epochs,
            mirror_gamma,
            ts_momentum,
        )
        self.final_errors = None
        self.grad_start_scores = torch.tensor(
            [1 / self.num_clients_subset] * self.num_clients_subset
        )
        self.surrogate_start_scores = torch.tensor(
            [1 / self.amount_of_clients] * self.amount_of_clients
        )
        self.best_surrogate = None

    def _init_trust_model(self):
        self.grad_model = AutoBANTModel2d(
            self.cfg,
            self.global_model.state_dict(),
            [self.client_gradients[rank] for rank in self.list_clients],
            self.device,
            init_trust_scores=None,
        )
        self.surrogate_model = AutoBANTModel2d(
            self.cfg,
            self.global_model.state_dict(),
            [
                self.final_errors[f"client {rank}"]
                for rank in range(self.amount_of_clients)
            ],
            self.device,
            init_trust_scores=None,
        )

    def _count_trust_score_manager(self, type):
        if type == "surrogate":
            self.trust_model = self.surrogate_model
            self.start_trust_scores = self.surrogate_start_scores
            ts = self._count_trust_score()
            self.surrogate_start_scores = self.start_trust_scores
        else:
            self.trust_model = self.grad_model
            self.start_trust_scores = self.grad_start_scores
            ts = self._count_trust_score()
            self.grad_start_scores = self.start_trust_scores
        return ts

    def evaluate_trust(self, model):
        model.eval()
        total_loss = 0
        len_accepted_batches = 0
        with torch.no_grad():
            for batch in self.trust_loader:
                _, (input, targets) = batch

                inp = input[0].to(self.device)
                targets = targets.to(self.device)

                outputs = model(inp)

                loss = self.criterion(outputs, targets)
                if torch.isnan(loss):
                    print("Nan in evaluate_trust!\n\n")
                    continue
                len_accepted_batches += 1
                total_loss += loss.item()

        return total_loss / len_accepted_batches if len_accepted_batches > 0 else 1234

    def _check_pi(self, ts):
        # In case when all clients of subset are maliciouses we need to reset ts to [0, ..., 0]
        # Now we just compare loss of (global_model + sum(ts * update)) with loss of (global_model)
        # We take clients even if the loss more than eps
        global_loss = self.evaluate_trust(self.global_model)

        saved_weights = copy.deepcopy(self.global_model.state_dict())
        aggregated_weights = self.global_model.state_dict()
        for it in range(len(self.list_clients)):
            rank = self.list_clients[it]
            grad_weight = ts[it]
            client_grad = self.client_gradients[rank]
            for key, grads in client_grad.items():
                aggregated_weights[key] = aggregated_weights[key] + grads.to(
                    self.device
                ) * grad_weight * self.gamma * (1 - self.theta)
        self.global_model.load_state_dict(aggregated_weights)

        new_loss = self.evaluate_trust(self.global_model)
        self.global_model.load_state_dict(saved_weights)
        print(f"global loss: {global_loss}, new loss: {new_loss}")
        # Hardcode eps == 0.05
        if global_loss < new_loss - 0.05:
            print("We don't apply ts(((((")
            return [0] * len(self.list_clients)
        else:
            print("Applying ts!!!!")
            return ts

    def _check_w(self, ts):
        # Here we evaluate w weights since weights computing is so unstable.
        # Now we just compare loss of newly calculated weights with the best weights.
        if self.best_surrogate is None:
            print("First init of w weights")
            self.best_surrogate = copy.deepcopy(ts)
            return ts
        saved_weights = copy.deepcopy(self.global_model.state_dict())
        aggregated_weights = self.global_model.state_dict()

        for rank in range(self.amount_of_clients):
            client_grad = self.client_gradients[rank]
            final_client_error = self.final_errors[f"client {rank}"]
            surrogate_weight = ts[rank]

            for key, grads in client_grad.items():
                aggregated_weights[key] = (
                    aggregated_weights[key]
                    + final_client_error[key]
                    * surrogate_weight
                    * self.gamma
                    * self.theta
                )

        self.global_model.load_state_dict(aggregated_weights)

        new_loss = self.evaluate_trust(self.global_model)
        self.global_model.load_state_dict(saved_weights)

        aggregated_weights = self.global_model.state_dict()

        for rank in range(self.amount_of_clients):
            client_grad = self.client_gradients[rank]
            final_client_error = self.final_errors[f"client {rank}"]
            surrogate_weight = self.best_surrogate[rank]

            for key, grads in client_grad.items():
                aggregated_weights[key] = (
                    aggregated_weights[key]
                    + final_client_error[key]
                    * surrogate_weight
                    * self.gamma
                    * self.theta
                )

        self.global_model.load_state_dict(aggregated_weights)

        best_surrogate_loss = self.evaluate_trust(self.global_model)
        print(f"from best: {best_surrogate_loss}, from autobant: {new_loss}")
        self.global_model.load_state_dict(saved_weights)

        if new_loss == 1234 or best_surrogate_loss == 1234:
            return self.best_surrogate
        if best_surrogate_loss <= new_loss:
            print("We apply best w weights")
            return self.best_surrogate
        else:
            print("Applying new w weights")
            self.best_surrogate = copy.deepcopy(ts)
            return ts
