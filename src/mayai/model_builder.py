import torch
from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm
from mayai.utils import commons
import logging as log

log.basicConfig(filename='model-builder.log', level=log.DEBUG, format='%(asctime)s %(message)s')


class ModelBuilder:

    def __init__(self, model, data, loss_fn, optimizer, checkpoint=None, model_path=None, scheduler=None,
                 device=commons.getDevice()):
        self.model = model
        self.lossFn = loss_fn
        self.optimizer = optimizer
        self.data = data
        self.checkpoint = checkpoint
        self.model_path = model_path

        self.trainer = EpochRunner(model=model, loss_fn=loss_fn, is_train=True, optimizer=optimizer,
                                   scheduler=optimizer if scheduler is None else scheduler)
        self.tester = EpochRunner(model=model, loss_fn=loss_fn, device=device, )

    def fit(self, epochs):
        log.info(f"Building model for {epochs}")

        train_metrices = []
        train_losses = []
        test_metrices = []
        test_losses = []
        learning_rates = []

        for e in range(0, epochs):
            print(f'\n\nEpoch: {e + 1}')
            log.info(f"Epoch {e}")

            learning_rate = self.optimizer.param_groups[0]['lr']
            learning_rates.append(learning_rate)

            log.info(f"Starting the training for epoch {e}")
            train_result = self.trainer.run_one_epoch(loader=self.data.train, epoch_num=e)
            train_losses.append((e, train_result.loss))
            print(f'Train Loss: {train_result.loss}, Learning Rate: {learning_rate}')

            if train_result.metric:
                train_metrices.append((e, train_result.metric))
                print(f'Train metrics: {train_result.metric}')

            log.info(f"End of the training for epoch {e}")

            # Test
            if (self.checkpoint is not None and e % self.checkpoint == 0) or (e == epochs - 1):

                log.info(f"Starting the testing for epoch {e}")
                print(f"Predicting on test set.")
                test_result = self.tester.run_one_epoch(loader=self.data.test, epoch_num=e)
                test_losses.append((e, test_result.loss))
                print(f'Test Loss: {test_result.loss}')

                if test_result.metric:
                    test_metrices.append((e, test_result.metric))
                    print(f'Test metrics: {test_result.metric}')
                log.info(f"End of the testing for epoch {e}")

                if self.model_path:
                    torch.save(self.model, self.model_path)
                    print(f"Saved the model to path:{self.model_path}")
                    log.info(f"Saved the model to path:{self.model_path}")

        return ModelBuildResult(train_metrices, train_losses, test_metrices, test_losses, learning_rates)


class EpochRunner:

    def __init__(self, model, loss_fn, optimizer=None, is_train=False, scheduler=None, metric_fn=None,
                 device=commons.getDevice()):
        self.optimizer = optimizer
        self.device = device
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.model = model
        self.metric_fn = metric_fn
        self.is_train = is_train
        self.name = "trainer" if is_train else "tester"

    def __run_one_batch(self, model, data, target):

        if self.optimizer:
            self.optimizer.zero_grad()
        output = model(data)
        loss = self.loss_fn(output, target)

        metric = None
        if self.metric_fn:
            metric = self.metric_fn(output, target)

        if self.is_train:
            loss.backward()
            self.optimizer.step()

        loss_value = loss.detach().item()
        return (loss_value, output.argmax(dim=1), metric)

    def run_one_epoch(self, loader, epoch_num):
        commons.cleanup()
        log.info(f"Finished cleanup for epoch {epoch_num}")

        if self.is_train:
            self.model.train()
        else:
            self.model.eval()

        pbar = tqdm(loader, ncols=1000)

        total_loss = 0
        sum_metric = 0

        num_batches = len(loader)
        log.info(f"Running {self.name} epoch: {epoch_num}")

        with torch.set_grad_enabled(self.is_train):
            for idx, (data, target) in enumerate(pbar):
                log.info(f"Obtained the data for batch:{idx}")
                data, target = data.to(self.device), target.to(self.device)

                log.info(f"Starting the {self.name} for batch:{idx}")
                (loss, prediction, metric) = self.__run_one_batch(self.model, data, target)
                log.info(f"End of the {self.name} for batch:{idx} with loss: {loss} and metric: {metric}")

                total_loss += loss
                if metric:
                    sum_metric += metric

                if self.is_train:
                    self.scheduler.step()
                    log.info(f"Scheduler step for the batch:{idx}")
                    lr = self.optimizer.param_groups[0]['lr']
                    pbar.set_description(desc=f'id={idx}\t Loss={loss}\t LR={lr}\t')
                    log.info(f"For train batch {idx} loss is {loss} and lr is {lr}")

                del loss, data, target
                log.info(f"Completed the {self.name} for batch:{idx}")

        return EpochResult(total_loss / num_batches, (sum_metric / num_batches) if self.metric_fn else None)


class EpochResult:

    def __init__(self, loss, metric=None):
        self.loss = loss
        self.metric = metric


class ModelBuildResult:

    def __init__(self, train_metrices, train_losses, test_metrices, test_losses, learning_rates=None):
        self.train_metrices = train_metrices
        self.train_losses = train_losses
        self.test_metrices = test_metrices
        self.test_losses = test_losses
        self.learning_rates = learning_rates
