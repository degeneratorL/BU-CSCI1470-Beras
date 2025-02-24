from collections import defaultdict

from beras.core import Diffable, Tensor

class GradientTape:

    def __init__(self):
        # Dictionary mapping the object id of an output Tensor to the Diffable layer it was produced from.
        self.previous_layers: defaultdict[int, Diffable | None] = defaultdict(lambda: None)

    def __enter__(self):
        # When tape scope is entered, all Diffables will point to this tape.
        if Diffable.gradient_tape is not None:
            raise RuntimeError("Cannot nest gradient tape scopes.")

        Diffable.gradient_tape = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # When tape scope is exited, all Diffables will no longer point to this tape.
        Diffable.gradient_tape = None

    def gradient(self, target: Tensor, sources: list[Tensor]) -> list[Tensor]:
        """
        Computes the gradient of the target tensor with respect to the sources.

        :param target: the tensor to compute the gradient of, typically loss output
        :param sources: the list of tensors to compute the gradient with respect to
        In order to use tensors as keys to the dictionary, use the python built-in ID function here: https://docs.python.org/3/library/functions.html#id.
        To find what methods are available on certain objects, reference the cheat sheet
        """

        ### TODO: Populate the grads dictionary with {weight_id, weight_gradient} pairs.

        queue = [target]                    ## Live queue; will be used to propagate backwards via breadth-first-search.
        grads = defaultdict(lambda: None)   ## Grads to be recorded. Initialize to None. Note: stores {id: list[gradients]}
        # Use id(tensor) to get the object id of a tensor object.
        # in the end, your grads dictionary should have the following structure:
        # {id(tensor): [gradient]}

        # What tensor and what gradient is for you to implement!
        # compose_input_gradients and compose_weight_gradients are methods that will be helpful
        grads[id(target)] = [None]

        # BFS or DFS: pop from queue, backtrack to layer inputs/weights
        while queue:
            out_tensor = queue.pop(0)
            layer = self.previous_layers[id(out_tensor)]
            if layer is None:
                # This tensor was not produced by a Diffable layer (maybe it's an input or constant)
                continue

            # Upstream gradients for `out_tensor`
            upstream_jacobians = grads[id(out_tensor)]

            # Compose with layer's input jacobians
            in_jac = layer.compose_input_gradients(upstream_jacobians)
            # For each input, accumulate the partial grads
            for inp, partial_grad in zip(layer.inputs, in_jac):
                if partial_grad is None:
                    continue
                if grads[id(inp)] is None:
                    grads[id(inp)] = [partial_grad]
                    queue.append(inp)  # We'll push inputs into queue to propagate further
                else:
                    grads[id(inp)].append(partial_grad)

            # Compose with layer's weight jacobians
            w_jac = layer.compose_weight_gradients(upstream_jacobians)
            for weight, partial_grad in zip(layer.weights, w_jac):
                if partial_grad is None:
                    continue
                if grads[id(weight)] is None:
                    grads[id(weight)] = [partial_grad]
                else:
                    grads[id(weight)].append(partial_grad)
            # We typically don't push weights into queue because weights generally don't have "previous layers."

        # Now we have a dictionary grads: { id(tensor): [grad_array_1, grad_array_2, ...] }
        # We sum them up for each id, then return them in the order of 'sources'.
        final_grads = []
        for src in sources:
            partials = grads[id(src)]
            if partials is None:
                # Means no gradient found => default to zeros of same shape as src
                final_grads.append(Tensor.zeros_like(src))
            else:
                # sum up all partial grads in partials
                accum = partials[0]
                for pg in partials[1:]:
                    accum = accum + pg
                final_grads.append(accum)

        return final_grads