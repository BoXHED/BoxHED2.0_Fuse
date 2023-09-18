import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput

import pdb

class LongformerClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.last_dense = nn.Linear(config.hidden_size, config.last_hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.last_hidden_size, config.num_labels)
        self.last_hidden_state = None

    def forward(self, hidden_states, return_embeddings, **kwargs):
        hidden_states = hidden_states[:, 0, :]  # take <s> token (equiv. to [CLS])
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.last_dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)

        self.last_hidden_state = hidden_states
        
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)

        if (return_embeddings):
            return (None, self.last_hidden_state)
        else:
            return (hidden_states, None)


class ClinicalLongformerForSequenceClassification(nn.Module):
    """
    Use an in-memory T5Encoder to do sequence classification"""
    def __init__(self, longformer, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.config = config

        self.encoder = longformer.double()  # already initialized model
        self.classifier = LongformerClassificationHead(config).double() # defined above

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_hidden_states=None,
        output_attentions=None,
        return_dict=None,
        return_embeddings=False
    ):
        surprise = [i for i in [head_mask, inputs_embeds, output_hidden_states, output_attentions, return_dict] if i != None]
        if surprise != []:
            print(f'surprise! these are not None: {surprise}')
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        assert(input_ids.device == attention_mask.device 
               and input_ids.device == self.encoder.device)
        
        # print("Initializing global attention on CLS token...")
        global_attention_mask = torch.zeros_like(input_ids)
        # global attention on cls token
        global_attention_mask[:, 0] = 1

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # global_attention_mask=global_attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # self.longformer(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     global_attention_mask=global_attention_mask,
        #     head_mask=head_mask,
        #     token_type_ids=token_type_ids,
        #     position_ids=position_ids,
        #     inputs_embeds=inputs_embeds,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )

        sequence_output = encoder_outputs[0]
        logits, classifier_last_hidden_state = self.classifier(
            sequence_output, return_embeddings)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                    # print("single_label_classification!!!")
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        
        assert(loss.item() != None)
        # print(f'loss: {loss.item()}')
        # loss = loss.unsqueeze(0) # FIXME only use this if training on multi-GPU

        if return_embeddings: 
            return classifier_last_hidden_state
        else:
            return SequenceClassifierOutput(           
                loss=loss,
                logits=logits,
                attentions=encoder_outputs.attentions,
            )
    # def half_forward(
    #     self,
    #     input_ids=None,
    #     attention_mask=None,
    #     head_mask=None,
    #     inputs_embeds=None,
    #     labels=None,
    #     output_hidden_states=None,
    #     output_attentions=None,
    #     return_dict=None,
    #     return_embeddings=False
    # ):
    #     r"""
    #     labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
    #         Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss),
    #         If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
    #     """
    #     return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    #     encoder_outputs = self.encoder(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         inputs_embeds=inputs_embeds,
    #         head_mask=head_mask,
    #         output_attentions=output_attentions,
    #         output_hidden_states=output_hidden_states,
    #         return_dict=return_dict,
    #     )

    #     return encoder_outputs
        