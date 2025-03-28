import math

import torch
from comet import download_model, load_from_checkpoint


class Evaluator:
    def __init__(self, trainer):
        self.trainer = trainer

    def eval(self):
        """
        Run the evaluator on the given dataset.
        """
        raise NotImplementedError


class PerplexityEvaluator(Evaluator):
    """
    Evaluator that computes the perplexity of a language model.
    """

    def eval(self):
        """Evaluate the model performance on the validation set (Perplexity)"""
        self.trainer.model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for X, Y, loss_mask in self.trainer.val_loader:
                X = X.to(self.trainer.args.device)
                Y = Y.to(self.trainer.args.device)
                loss_mask = loss_mask.to(self.trainer.args.device)

                res = self.trainer.model(X)
                loss = self.trainer.loss_fct(res.logits.view(-1, res.logits.size(-1)), Y.view(-1)).view(Y.size())
                loss = (loss * loss_mask).sum()  # Sum loss for valid tokens
                total_loss += loss.item()
                total_tokens += loss_mask.sum().item()  # Count valid tokens

        # Calculate perplexity based on total tokens
        perplexity = math.exp(total_loss / total_tokens)
        self.trainer.log(f"Validation Perplexity: {perplexity:.2f}")


class CometEvaluator(Evaluator):
    """
    Evaluator that computes the COMET score of a language model.
    """

    def __init__(self, trainer):
        super().__init__(trainer)
        self.comet_model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))

    def _build_data(
        self,
        sources: list[str],
        references: list[str],
        translations: list[str],
    ) -> list[dict[str, str]]:
        data: list[dict[str, str]] = []
        for source, reference, translation in zip(sources, references, translations):
            data.append(
                {
                    "src": source,
                    "mt": translation,
                    "ref": reference,
                }
            )
        return data

    def _get_samples(self):
        # Get subset from the trainer
        subset = self.trainer.val_loader.dataset

        # Get indices
        indices = subset.indices

        # Get samples
        samples = [subset.dataset.samples[indices[i]] for i in range(len(indices))]

        return samples

    def _get_sources(self):
        return self.trainer.val_loader.dataset.dataset.get_sources(self._get_samples())

    def _get_references(self):
        return self.trainer.val_loader.dataset.dataset.get_references(self._get_samples())

    def _get_messages_lst(self):
        for sample in self._get_samples():
            yield self.trainer.val_loader.dataset.dataset.extract_messages(sample)

    def eval(self):
        """
        Evaluate the model on the given dataset and return the COMET score.
        """
        # Collect sources from the dataset
        sources: list[str] = self._get_sources()

        # Collect references from the dataset
        references: list[str] = self._get_references()

        # Collect predictions from the model
        predictions: list[str] = self.trainer.get_predictions(self._get_messages_lst())

        # Build data for COMET evaluation
        data: list[dict[str, str]] = self._build_data(sources, references, predictions)

        # Evaluate using COMET
        model_output = self.comet_model.predict(data, batch_size=32, gpus=1)

        # Log the scores
        self.trainer.log(f"COMET Score: {model_output.system_score:.2f}")
