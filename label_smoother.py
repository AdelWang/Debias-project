# The overlap label smoothing mechanism
def compute_loss(self, model, inputs, classifier=None, model_extra=None, return_outputs=False):
        """
        How the loss is computed by Trainer_self. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        loss_pred = 0.
        loss_sim = None
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if "overlap" in inputs:
            overlap = inputs.pop("overlap")
            overlap_target = 0.5 * torch.ones_like(overlap)
        else:
            overlap = None
        labels = inputs.pop("labels")
        smoother_labels = F.one_hot(labels, num_classes=model.config.num_labels).float()
        for index in range(len(smoother_labels)):
            if smoother_labels[index,1] == 1:
                smoother_labels[index] += 0.45 * overlap[index] / (model.config.num_labels - 1)
                smoother_labels[index,1] -= 0.45 * overlap[index] * (1 + 1/(model.config.num_labels - 1))
        outputs = model(**inputs,labels=smoother_labels)
