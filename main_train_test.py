import torch
import torch.utils.data as data
from data_loader import DataSet
from models.MbPAplusplus import ReplayMemory, MbPAplusplus
import transformers
# from tqdm import trange, tqdm
from tqdm.notebook import trange, tqdm
import time
import copy
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

# Use cudnn backends instead of vanilla backends when the input sizes
# are similar so as to enable cudnn which will try to find optimal set
# of algorithms to use for the hardware leading to faster runtime.
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

LEARNING_RATE = 3e-5
MODEL_NAME = 'MbPA++'
# Due to memory restraint, we sample only 64 examples from
# stored memory after every 6400(1% replay rate) new examples seen
# as opposed to 100 suggested in the paper. The sampling is done after
# performing 200 steps(6400/32).
REPLAY_FREQ = 201

class MbPA_Experiment():

    def __init__(self, batch_size=32, mode='train', order=1, epochs=1, model_path=None, memory_path = None, save_interval=4000):

        self.use_cuda = True if torch.cuda.is_available() else False
        self.batch_size = batch_size
        self.mode = mode
        self.order = order
        self.epochs = epochs
        self.model_path = model_path
        self.memory_path = memory_path
        self.save_interval = save_interval

        if self.mode == 'train':
            self.model = MbPAplusplus()
            self.memory = ReplayMemory()
            self.train()

        if self.mode == 'test':
            model_state = torch.load(
                self.model_path)
            self.model = MbPAplusplus(model_state=model_state)
            buffer = {}
            with open(self.memory_path, 'rb') as f:
                buffer = pickle.load(f)
            self.memory = ReplayMemory(buffer=buffer)
            self.test()



    def train(self):
        """
        Train function
        """
        workers = 0
        if self.use_cuda:
            self.model.cuda()
            # Number of workers should be 4*num_gpu_available
            # https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/5
            workers = 4
        # time at the start of training
        start = time.time()

        train_data = DataSet(self.order, split='train')
        train_sampler = data.SequentialSampler(train_data)
        train_dataloader = data.DataLoader(
            train_data, sampler=train_sampler, batch_size=self.batch_size, num_workers=workers)
        param_optimizer = list(self.model.classifier.named_parameters())
        # parameters that need not be decayed
        no_decay = ['bias', 'gamma', 'beta']
        # Grouping the parameters based on whether each parameter undergoes decay or not.
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.0}]
        optimizer = transformers.AdamW(
            optimizer_grouped_parameters, lr=LEARNING_RATE)

        # Store our loss and accuracy for plotting
        train_loss_set = []
        # trange is a tqdm wrapper around the normal python range
        # for epoch in tnrange(self.epochs, desc="Epoch"):
        for epoch in trange(self.epochs, desc='Epochs'):
        # for epoch in range(self.epochs):
            # Training begins
            print("Training begins")
            # Set our model to training mode (as opposed to evaluation mode)
            self.model.classifier.train()
            # Tracking variables
            tr_loss = 0
            nb_tr_examples, nb_tr_steps, num_curr_exs = 0, 0, 0
            # Train the data for one epoch
            for step, batch in enumerate(tqdm(train_dataloader, desc='Batch')):
            # for step, batch in enumerate(train_dataloader):
                # Release file descriptors which function as shared
                # memory handles otherwise it will hit the limit when
                # there are too many batches at dataloader
                batch_cp = copy.deepcopy(batch)
                del batch
                # Perform sparse experience replay after every REPLAY_FREQ steps
                if (step+1) % REPLAY_FREQ == 0:
                    # sample 64 examples from memory
                    content, attn_masks, labels = self.memory.sample(sample_size=self.batch_size)
                    if self.use_cuda:
                        content = content.cuda()
                        attn_masks = attn_masks.cuda()
                        labels = labels.cuda()
                    # Clear out the gradients (by default they accumulate)
                    optimizer.zero_grad()
                    # Forward pass
                    loss, _ = self.model.classify(content, attn_masks, labels)
                    train_loss_set.append(loss.item())
                    # Backward pass
                    loss.backward()
                    # Update parameters and take a step using the computed gradient
                    optimizer.step()

                    # Update tracking variables
                    tr_loss += loss.item()
                    nb_tr_examples += content.size(0)
                    nb_tr_steps += 1

                    del content
                    del attn_masks
                    del labels
                    del loss

                if (step+1) % self.save_interval == 0:
                    print('Saving checkpoint at step ' , str(step+1))
                    model_dict = self.model.save_state()
                    self.save_checkpoint(model_dict, epoch+1, iteration=str(step+1))

                # Unpacking the batch items
                content, attn_masks, labels = batch_cp
                content = content.squeeze(1)
                attn_masks = attn_masks.squeeze(1)
                labels = labels.squeeze(1)
                # number of examples in the current batch
                num_curr_exs = content.size(0)
                # Place the batch items on the appropriate device: cuda if avaliable
                if self.use_cuda:
                    content = content.cuda()
                    attn_masks = attn_masks.cuda()
                    labels = labels.cuda()
                # Clear out the gradients (by default they accumulate)
                optimizer.zero_grad()
                # Forward pass
                loss, _ = self.model.classify(content, attn_masks, labels)
                train_loss_set.append(loss.item())
                # Get the key representation of documents
                keys = self.model.get_keys(content, attn_masks)
                # Push the examples into the replay memory
                self.memory.push(keys.cpu().numpy(), (content.cpu().numpy(),
                                                attn_masks.cpu().numpy(), labels.cpu().numpy()))
                # delete the batch data to freeup gpu memory
                del keys
                del content
                del attn_masks
                del labels
                # Backward pass
                loss.backward()
                # Update parameters and take a step using the computed gradient
                optimizer.step()
                # Update tracking variables
                tr_loss += loss.item()
                nb_tr_examples += num_curr_exs
                nb_tr_steps += 1

            now = time.time()
            print("Train loss: {}".format(tr_loss/nb_tr_steps))
            print("Time taken till now: {} hours".format((now-start)/3600))
            model_dict = self.model.save_state()
            self.save_checkpoint(model_dict, epoch+1, memory=True)

        self.save_trainloss(train_loss_set)


    def save_checkpoint(self, model_dict, epoch, iteration='', memory=None):
        """
        Function to save a model checkpoint to the specified location
        """
        base_loc = 'model_checkpoints'
        if not os.path.exists(base_loc):
            os.mkdir('model_checkpoints')

        checkpoints_dir = os.path.join(base_loc, MODEL_NAME)
        if not os.path.exists(checkpoints_dir):
            os.mkdir(checkpoints_dir)
        checkpoints_file = 'classifier_order_' + \
            str(self.order) + '_epoch_'+str(epoch)+'_'+iteration+'.pth'
        torch.save(model_dict, os.path.join(checkpoints_dir, checkpoints_file))
        memory_file = 'order_'+str(self.order)+'_epoch_'+str(epoch)+'_'+iteration+'.pkl'
        if memory:
            with open(os.path.join(checkpoints_dir,memory_file), 'wb') as f:
                pickle.dump(self.memory.memory, f)


    def calc_correct(self, preds, labels):
        """
        Function to calculate the accuracy of our predictions vs labels
        """
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat)


    def test(self):
        """
        evaluate the model for accuracy
        """
        # time at the start of validation
        start = time.time()
        if self.use_cuda:
            self.model.cuda()

        test_data = DataSet(self.order, split='test')
        test_dataloader = data.DataLoader(
            test_data, shuffle=True, batch_size=64, num_workers=4)

        # Tracking variables
        total_correct, tmp_correct, t_steps = 0, 0, 0

        print("Validation step started...")
        for batch in tqdm(test_dataloader, desc='Batch'):
            batch_cp = copy.deepcopy(batch)
            del batch
            contents, attn_masks, labels = batch_cp
            if self.use_cuda:
                contents = contents.squeeze(1).cuda()
                attn_masks = attn_masks.squeeze(1).cuda()
            keys = self.model.get_keys(contents, attn_masks)
            retrieved_batches = self.memory.get_neighbours(keys.cpu().numpy())
            del keys
            ans_logits = []
            # Iterate over the test batch to calculate label for each document(i.e,content)
            # and store them in a list for comparision later
            for content, attn_mask, (rt_contents, rt_attn_masks, rt_labels) in tqdm(zip(contents, attn_masks, retrieved_batches), total=len(contents), desc='Refit' , leave=False):
                if self.use_cuda:
                    rt_contents = rt_contents.cuda()
                    rt_attn_masks = rt_attn_masks.cuda()
                    rt_labels = rt_labels.cuda()

                logits = self.model.infer(content, attn_mask,
                                    rt_contents, rt_attn_masks, rt_labels)

                ans_logits.append(logits.cpu().numpy())
            # Dropping the 1 dim to match the logits' shape
            # shape : (batch_size,num_labels)
            labels = labels.squeeze(1).numpy()
            tmp_correct = self.calc_correct(np.asarray(ans_logits), labels)
            # del labels
            total_correct += tmp_correct
            t_steps += len(labels.flatten())
        end = time.time()
        print("Time taken for validation {} minutes".format((end-start)/60))
        print("Validation Accuracy: {}".format(total_correct/t_steps))


    def save_trainloss(self, train_loss_set):
        """
        Function to save the image of training loss v/s iterations graph
        """
        plt.figure(figsize=(15, 8))
        plt.title("Training loss")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.plot(train_loss_set)
        base_loc = './loss_images'
        if not os.path.exists(base_loc):
            os.mkdir(base_loc)

        image_dir = os.path.join(base_loc,MODEL_NAME)
        if not os.path.exists(image_dir):
            os.mkdir(image_dir)

        plt.savefig(os.path.join(image_dir, 'order_'+str(self.order)+'_train_loss.png'))