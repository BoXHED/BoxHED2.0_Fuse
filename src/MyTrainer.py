from transformers import Trainer
from transformers.modeling_utils import unwrap_model
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

class MyTrainer(Trainer):
    def __init__(self, model, args, train_dataset=None, eval_dataset=None,
                 tokenizer=None, compute_metrics=None):
        super().__init__(model, args, train_dataset=train_dataset,
                         eval_dataset=eval_dataset, tokenizer=tokenizer,
                         compute_metrics=compute_metrics)
        print(self)
        # Add any additional setup code specific to your OwnTrainer class here

    def _save_checkpoint(self, model, trial, metrics=None):
        # Insert your own behavior here to save the checkpoint
        # You can use the provided `model` argument to access the model
        # You can use the `trial` argument if you're using Optuna for hyperparameter tuning
        # You can use the `metrics` argument to access the evaluation metrics
        
        # Call the original function
        super()._save_checkpoint(model, trial, metrics)

        model_out = unwrap_model(model)
        print(f"I have a model: {model.__class__}")

        epoch = int(round(self.state.epoch))
        checkpoint_path = f"{self.args.output_dir}/model_checkpoint_epoch{epoch}.pt"
        print(f"I want to save to: {checkpoint_path}")
        torch.save(model_out, checkpoint_path)  # Save model state_dict to .pt file
    
    # def predict(self, test_dataset, ignore_keys=None):
    #     self.evaluation_loop
    #     output = eval_loop(
    #         test_dataloader, description="Prediction", ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
    #     )

# class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        breakpoint()
        outputs = model.forward(**inputs)

        logits = outputs.get('logits')
        # loss = outputs.get('loss')

        if labels is not None:
            # if self.config.problem_type is None:
            #     if self.num_labels == 1:
            #         self.config.problem_type = "regression"
            #     elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
            #         self.config.problem_type = "single_label_classification"
            #     else:
            #         self.config.problem_type = "multi_label_classification"

            # if self.config.problem_type == "regression":
            loss_fct = MSELoss()
            num_labels = model.config.num_labels
            if num_labels == 1:
                loss = loss_fct(logits.float().view(-1, self.model.config.num_labels), 
                                labels.float().view(-1, self.model.config.num_labels))
                breakpoint()
                print(f'loss: {loss}')
            # else:
            #     loss = loss_fct(logits, labels)
            # elif self.config.problem_type == "single_label_classification":
            #     loss_fct = CrossEntropyLoss()
            #     loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            # elif self.config.problem_type == "multi_label_classification":
            #     loss_fct = BCEWithLogitsLoss()
            #     loss = loss_fct(logits, labels)

        # num_labels = self.model.config.num_labels
        # if num_labels > 1:
        #     labels_onehot = torch.zeros(labels.size(0), num_labels, device=labels.device)  # Create an empty tensor for one-hot encoding
        #     labels_onehot.scatter_(1, labels.unsqueeze(1), 1)  # Perform one-hot encoding

        # loss_fct = nn.BCEWithLogitsLoss()
#         print(f'logits: {logits},\
# labels: {labels}')
#         print(f'logits view: {logits.view(-1, self.model.config.num_labels)},\
# labels view: {labels.float().view(-1, self.model.config.num_labels)}')
        # loss = loss_fct(logits.view(-1, self.model.config.num_labels),
        #                 labels_onehot.float().view(-1, self.model.config.num_labels))
        
        return (loss, outputs) if return_outputs else loss





# def _save_checkpoint_OLD(self, model, trial, metrics=None):
#         # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
#         # want to save except FullyShardedDDP.
#         # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

#         # Save model checkpoint
#         checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

#         if self.hp_search_backend is None and trial is None:
#             self.store_flos()

#         run_dir = self._get_output_dir(trial=trial)
#         output_dir = os.path.join(run_dir, checkpoint_folder)
#         self.save_model(output_dir, _internal_call=True)
#         if self.deepspeed:
#             # under zero3 model file itself doesn't get saved since it's bogus! Unless deepspeed
#             # config `stage3_gather_16bit_weights_on_model_save` is True
#             self.deepspeed.save_checkpoint(output_dir)

#         # Save optimizer and scheduler
#         if self.sharded_ddp == ShardedDDPOption.SIMPLE:
#             self.optimizer.consolidate_state_dict()

#         if self.fsdp:
#             # FSDP has a different interface for saving optimizer states.
#             # Needs to be called on all ranks to gather all states.
#             # full_optim_state_dict will be deprecated after Pytorch 2.2!
#             full_osd = self.model.__class__.full_optim_state_dict(self.model, self.optimizer)

#         if is_torch_tpu_available():
#             xm.rendezvous("saving_optimizer_states")
#             xm.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
#             with warnings.catch_warnings(record=True) as caught_warnings:
#                 xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
#                 reissue_pt_warnings(caught_warnings)
#         elif is_sagemaker_mp_enabled():
#             opt_state_dict = self.optimizer.local_state_dict(gather_if_shard=False)
#             smp.barrier()
#             if smp.rdp_rank() == 0 or smp.state.cfg.shard_optimizer_state:
#                 smp.save(
#                     opt_state_dict,
#                     os.path.join(output_dir, OPTIMIZER_NAME),
#                     partial=True,
#                     v3=smp.state.cfg.shard_optimizer_state,
#                 )
#             if self.args.should_save:
#                 with warnings.catch_warnings(record=True) as caught_warnings:
#                     torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
#                 reissue_pt_warnings(caught_warnings)
#                 if self.do_grad_scaling:
#                     torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))
#         elif self.args.should_save and not self.deepspeed:
#             # deepspeed.save_checkpoint above saves model/optim/sched
#             if self.fsdp:
#                 torch.save(full_osd, os.path.join(output_dir, OPTIMIZER_NAME))
#             else:
#                 torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))

#             with warnings.catch_warnings(record=True) as caught_warnings:
#                 torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
#             reissue_pt_warnings(caught_warnings)
#             if self.do_grad_scaling:
#                 torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))

#         # Determine the new best metric / best model checkpoint
#         if metrics is not None and self.args.metric_for_best_model is not None:
#             metric_to_check = self.args.metric_for_best_model
#             if not metric_to_check.startswith("eval_"):
#                 metric_to_check = f"eval_{metric_to_check}"
#             metric_value = metrics[metric_to_check]

#             operator = np.greater if self.args.greater_is_better else np.less
#             if (
#                 self.state.best_metric is None
#                 or self.state.best_model_checkpoint is None
#                 or operator(metric_value, self.state.best_metric)
#             ):
#                 self.state.best_metric = metric_value
#                 self.state.best_model_checkpoint = output_dir

#         # Save the Trainer state
#         if self.args.should_save:
#             self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

#         # Save RNG state in non-distributed training
#         rng_states = {
#             "python": random.getstate(),
#             "numpy": np.random.get_state(),
#             "cpu": torch.random.get_rng_state(),
#         }
#         if torch.cuda.is_available():
#             if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
#                 # In non distributed, we save the global CUDA RNG state (will take care of DataParallel)
#                 rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
#             else:
#                 rng_states["cuda"] = torch.cuda.random.get_rng_state()

#         if is_torch_tpu_available():
#             rng_states["xla"] = xm.get_rng_state()

#         # A process can arrive here before the process 0 has a chance to save the model, in which case output_dir may
#         # not yet exist.
#         os.makedirs(output_dir, exist_ok=True)

#         if self.args.world_size <= 1:
#             torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
#         else:
#             torch.save(rng_states, os.path.join(output_dir, f"rng_state_{self.args.process_index}.pth"))

#         if self.args.push_to_hub:
#             self._push_from_checkpoint(output_dir)

#         # Maybe delete some older checkpoints.
#         if self.args.should_save:
#             self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)