# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for Simple Continual Learning datasets
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------

import random
import os
import torch
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms

from timm.data import create_transform

from continual_datasets.continual_datasets import *

import utils
from copy import deepcopy

class Lambda(transforms.Lambda):
    def __init__(self, lambd, nb_classes):
        super().__init__(lambd)
        self.nb_classes = nb_classes
    
    def __call__(self, img):
        return self.lambd(img, self.nb_classes)

def target_transform(x, nb_classes):
    return x + nb_classes

def build_continual_dataloader(args):
    dataloader = list()
    class_mask = list() if args.task_inc or args.train_mask else None

    transform_train = build_transform(True, args)
    transform_val = build_transform(False, args)

    if args.dataset.startswith('Split-'):
        dataset_train, dataset_val = get_dataset(args.dataset.replace('Split-',''), transform_train, transform_val, args)

        args.nb_classes = len(dataset_val.classes)

        splited_dataset, class_mask = split_single_dataset(dataset_train, dataset_val, args)
    else:
        if args.dataset == '5-datasets':
            dataset_list = ['SVHN', 'MNIST', 'CIFAR10', 'NotMNIST', 'FashionMNIST']
        if args.dataset in ['vtab-1k', 'overlapping-vtab-1k']:
            vtab_path = args.data_path # 
            dataset_list = os.listdir(vtab_path)
            print(dataset_list)
            if len(args.vtab_datasets) > 0 :
                dataset_list = args.vtab_datasets
                
        else:
            dataset_list = args.dataset.split(',')
        
        if args.shuffle:
            random.shuffle(dataset_list)

        print(dataset_list)
        
        if args.dataset in ['vtab-1k', 'overlapping-vtab-1k'] and args.milder_vtab:
            ## vtab configuration ## 
            
            # Natural 7: Caltech101,CIFAR100,DTD,Flowers102,Pets,Sun397,and SVHN.
            # Structured 8: Clevr2, dSprites2, SmallNORB2, DMLab, KITTI
            # Specialized 4: Resisc45 and EuroSAT; PatchCamelyon and Diabetic Retinopathy
            vtabgroup2dts = {'Natural':['caltech101','cifar','dtd','oxford_flowers102','oxford_iiit_pet','sun397','svhn'],
                            'Structured':['clevr_dist','clevr_count','dsprites_ori','dsprites_loc',
                                        'smallnorb_ele','smallnorb_azi', 'kitti', 'dmlab'],
                            'Specialized':['patch_camelyon','diabetic_retinopathy','eurosat','resisc45',],}

            if args.clustering_based_vtabgroups:
                vtabgroup2dts = {'0':['smallnorb_azi', 'smallnorb_ele', 'svhn', 'cifar'],
                                 '1':['oxford_iiit_pet', 'oxford_flowers102', 'caltech101'],
                                 '2':['dtd', 'resisc45', 'eurosat', 'dmlab', 'kitti', 'sun397'],
                                 '3':['patch_camelyon', 'diabetic_retinopathy'],
                                 '4':['dsprites_ori', 'dsprites_loc'],
                                 '5':['clevr_count', 'clevr_dist'],}

            if args.vtab_group_order:
                dataset_list = []
                for k in list(vtabgroup2dts.keys()):
                    dataset_list = dataset_list + vtabgroup2dts[k]
                print(dataset_list)
                args.shuffle_overlaps = False

            # subtract no-mild tasks from vtabgroup2dts
            if args.no_mild_tasks:
                no_mild_ts = random.sample(dataset_list, k=args.num_no_mild_tasks)
                print('no_mild_tasks', no_mild_ts)
                for task in no_mild_ts:
                    for group, dts  in vtabgroup2dts.items():
                        if task in dts:
                            vtabgroup2dts[group].remove(task)

            if args.milder_by_alldts:
                all_dts = vtabgroup2dts['Natural']+vtabgroup2dts['Structured']+vtabgroup2dts['Specialized']
                vtabgroup2dts = {'Natural':all_dts,'Structured':all_dts,'Specialized':all_dts,}

            dt2dtids = {dt:dt_id for dt_id,dt in enumerate(dataset_list)} # e.g., {dtd:0, ...}

            if not args.clustering_based_vtabgroups:
                vtabgroup2dtids = {'Natural':[ dt2dtids[dt] for dt in vtabgroup2dts['Natural'] if dt in dt2dtids ],
                                    'Structured':[ dt2dtids[dt] for dt in vtabgroup2dts['Structured'] if dt in dt2dtids ],
                                    'Specialized':[ dt2dtids[dt] for dt in vtabgroup2dts['Specialized'] if dt in dt2dtids ],}
            else: # args.clustering_based_vtabgroups
                vtabgroup2dtids = {group:[ dt2dtids[dt] for dt in dts if dt in dt2dtids ] \
                                    for group, dts in vtabgroup2dts.items()}
                

            dtid2groupdtids = dict() # e.g., {0:[0,3,4], ...}
            for dtid in list(dt2dtids.values()):
                in_group = False
                for vtab_g, dtids in vtabgroup2dtids.items():
                    if dtid in dtids:
                        dtid2groupdtids[dtid] = dtids
                        in_group = True

                if not in_group: # spcf task: no mild task
                    dtid2groupdtids[dtid] = [dtid]
            
        args.nb_classes = 0

    if args.dataset in ['overlapping-vtab-1k', 'overlapping-Split-CIFAR100', 'overlapping-Split-Imagenet-R']:
        overlapping_inds = torch.randperm(args.num_tasks)[:args.num_overlapping_tasks]
        if len(args.overlap_datasets) > 0 :
            overlapping_inds = torch.tensor([i for i,dt in enumerate(dataset_list) if dt in args.overlap_datasets ])
            
        print('overlapping_inds', overlapping_inds)
    else: 
        overlapping_inds = None
        
    # for making milder_vtab
    datasets = []
    
    for i in range(args.num_tasks):
        if args.dataset.startswith('Split-'):
            dataset_train, dataset_val = splited_dataset[i]

        else:
            dataset_train, dataset_val = get_dataset(dataset_list[i], transform_train, transform_val, args)

            transform_target = Lambda(target_transform, args.nb_classes)

            if class_mask is not None:
                if args.dataset in ['vtab-1k', 'overlapping-vtab-1k']:
                    exposed_cls = dataset_train.classes.union(dataset_val.classes)
                    exposed_cls = list(range( min(exposed_cls), max(exposed_cls)+1 ))
                else:
                    exposed_cls = dataset_train.classes
                class_mask.append([i + args.nb_classes for i in range(len(exposed_cls))])
                args.nb_classes += len(exposed_cls)
                print('args.nb_classes', args.nb_classes)
                
            if not args.task_inc: # set to true
                dataset_train.target_transform = transform_target
                dataset_val.target_transform = transform_target
        
        if args.small_dataset:
            num_subset = int(len(dataset_train) * args.small_dataset_scale)
            subset_inds = torch.randperm(len(dataset_train))[:num_subset]
            dataset_train = Subset(dataset_train, subset_inds)
            
        if overlapping_inds is None:
            if args.distributed and utils.get_world_size() > 1:
                num_tasks = utils.get_world_size()
                global_rank = utils.get_rank()

                
                sampler_train = torch.utils.data.DistributedSampler(
                        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
                sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            else:
                sampler_train = torch.utils.data.RandomSampler(dataset_train)
                sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            
            datasets += [{'data_id':i ,'data':dataset_train}] # for making milder_vtab
            data_loader_train = torch.utils.data.DataLoader(
                dataset_train, sampler=sampler_train,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem, 
                # drop_last = True
            )

            data_loader_val = torch.utils.data.DataLoader(
                dataset_val, sampler=sampler_val,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
            )

            dataloader.append({'train': data_loader_train, 'val': data_loader_val})
            
        else: # add the dts as many as num_overlappings
            # args: num_overlaps, overlap_dataset_scale, num_overlapping_tasks, shuffle_overlaps
            n_overlaps = args.num_overlaps if i in overlapping_inds else 1 
                
            for _ in range(n_overlaps):
                datasets += [{'data_id':i ,'data':dataset_train}] # for making milder_vtab

                num_subset = int(len(dataset_train) * args.overlap_dataset_scale)
                subset_inds = torch.randperm(len(dataset_train))[:num_subset]
                sub_dataset_train = Subset(dataset_train, subset_inds)
                
                if args.distributed and utils.get_world_size() > 1:
                    num_tasks = utils.get_world_size()
                    global_rank = utils.get_rank()

                    
                    sampler_train = torch.utils.data.DistributedSampler(
                            sub_dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
                    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
                else:
                    sampler_train = torch.utils.data.RandomSampler(sub_dataset_train)
                    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
                
                data_loader_train = torch.utils.data.DataLoader(
                    sub_dataset_train, sampler=sampler_train,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    pin_memory=args.pin_mem, 
                    # drop_last = True
                )

                data_loader_val = torch.utils.data.DataLoader(
                    dataset_val, sampler=sampler_val,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    pin_memory=args.pin_mem,
                )

                dataloader.append({'train': data_loader_train, 'val': data_loader_val, 'task':i})
            
            for _ in range(n_overlaps-1):
                class_mask.append(class_mask[-1])
    
    ### make datasets blurry using datasets within the same group ###################################################
    if args.milder_vtab: 
        ## assume the n_overlaps = 1
        # update datasets
        new_blurry_train_datasets = []
        for dt in datasets: # #datasets = n_overlaps x num_tasks
            # - get other dtids within the same group
            data_id, data = dt['data_id'], dt['data'] # data: VTAB dataset
            group_dt_ids = deepcopy(dtid2groupdtids[data_id])
            group_dt_ids.remove(data_id)
            group_datas = [datasets[other_dtid * n_overlaps]['data'] for other_dtid in group_dt_ids]
            group_datas_exist = False if len(group_datas) == 0 else True

            print('data_id: ', data_id, 'other_data_id: ', group_dt_ids,)
            print('100samples of data', [data[_][1] for _ in range(50)])
            # - get (1-n)% of this dt & n% of other dts; n: overlap_similarity
            n_samples_data = len(data.samples)
            if group_datas_exist:
                keep_samples = int( n_samples_data * ((100-args.overlap_similarity)/100) ) # (1-n)%
                update_samples = n_samples_data - keep_samples # n%
                update_samples_per_other_dt = int(update_samples//len(group_datas)) # n%/len(other_dts)
                update_samples_list = [update_samples_per_other_dt]*len(group_datas)
                update_samples_list[-1] = update_samples_list[-1] + (update_samples - sum(update_samples_list))
                print('total samplings: ', n_samples_data)
                print('n_sampling : ', [keep_samples]+update_samples_list)
            else:
                keep_samples = n_samples_data
                update_samples_list = []
            # construct new samples
            # -- select indices (for removing&replacing) from each dt
            keep_indices = torch.randperm(n_samples_data)[:keep_samples]
            new_samples = [(data.samples[ind][0], data[ind][1]) for ind in keep_indices] 
            # (path, label); must use indexing (__getitem__) for getting labels (to use `target_transform`)
            for other_dt, n_replace_samples_per_dt in zip(group_datas, update_samples_list):
                pick_indices = torch.randperm(len(other_dt))[:n_replace_samples_per_dt]
                new_samples += [(other_dt.samples[ind][0], other_dt[ind][1]) for ind in pick_indices] # (path, label)
                
            clone_data = deepcopy(data)
            # update samples & nullify target_transform for new datasets
            clone_data.samples = new_samples
            clone_data.target_transform = None
            new_blurry_train_datasets += [clone_data]

            random_inds = torch.randperm(len(data))[:20]
            print('20 labels) data.samples', 
                  [data[_][1] for _ in random_inds], )
            print('20 labels) clone_data.samples', 
                  [clone_data[_][1] for _ in random_inds] )
            
        # update dataloaders & class_mask
        for dt_i, new_dt in enumerate(new_blurry_train_datasets):
            if args.distributed and utils.get_world_size() > 1:
                num_tasks = utils.get_world_size()
                global_rank = utils.get_rank()

                sampler_train = torch.utils.data.DistributedSampler(
                        new_dt, num_replicas=num_tasks, rank=global_rank, shuffle=True)
                # sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            else:
                sampler_train = torch.utils.data.RandomSampler(new_dt)
                # sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            
            data_loader_train = torch.utils.data.DataLoader(
                new_dt, sampler=sampler_train,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem, 
                # drop_last = True
            )
            data_loader_val = dataloader[dt_i]['val']

            dataloader[dt_i] = {'train': data_loader_train, 'val': data_loader_val, 'task':dt_i}
            new_class_mask = [label for path, label in new_dt.samples] # exposed_classes
            class_mask[dt_i] = new_class_mask
    ### make datasets blurry using datasets within the same group ###################################################


    if (args.dataset in ['overlapping-vtab-1k',]) and (args.shuffle_overlaps): # shuffle or not        
        rand_inds = torch.tensor(args.shuffle_overlaps_inds) if len(args.shuffle_overlaps_inds)>0 else torch.randperm(len(dataloader))
        print('rand_inds', rand_inds)
        if args.save_warmup_prompts : # save task config
            Path(os.path.join(args.output_dir, 'task_config')).mkdir(parents=True, exist_ok=True)
            task_config_path = os.path.join(args.output_dir, 'task_config/task_order.pth')
            if args.data_driven_evolve and args.uni_or_specific:
                utils.save_on_master(rand_inds, task_config_path)
        dataloader = [ dataloader[i.item()] for i in rand_inds ]
        class_mask = [ class_mask[i.item()] for i in rand_inds ]
        # datasets = [ datasets[i.item()] for i in rand_inds ]
    if args.dataset in ['overlapping-vtab-1k',]:
        print("Before) num_tasks: ", args.num_tasks)
        args.num_tasks = args.num_tasks + (args.num_overlapping_tasks) * (args.num_overlaps-1)
        print("After) num_tasks: ", args.num_tasks)
        print([dataloader[i]['task'] for i in range(len(dataloader))])
        
    return dataloader, class_mask


class MyCIFAR100(datasets.CIFAR100):
    """
    Overrides the CIFAR100 dataset to change the getitem function.
    """
    def __init__(self, root, train=True, transform=None,download=False,noisy_labels=False) -> None:
        # self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        self.original_targets = None
        self.noisy_labels = noisy_labels
        super(MyCIFAR100, self).__init__(root, train=train, download=download, transform=transform,)

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.original_targets is not None:
            original_trg = self.original_targets[index]
            noise = 0 if original_trg == target else 1
            # print(target, self.original_targets[index], noise)
        else:
            noise = 0

        if self.noisy_labels:
            return img, target, noise
        else:
            return img, target


def get_dataset(dataset, transform_train, transform_val, args,):
    if args.dataset in ['vtab-1k', 'overlapping-vtab-1k']:
        datset_path = os.path.join(args.data_path, dataset) # vtab-path
        dataset_train = VTAB(root=datset_path, train=True, transform=transform_train)
        dataset_val = VTAB(root=datset_path, train=False, transform=transform_val)
        
        return dataset_train, dataset_val
    
    if dataset == 'CIFAR100':
        # dataset_train = datasets.CIFAR100(args.data_path, train=True, download=True, transform=transform_train)
        # dataset_val = datasets.CIFAR100(args.data_path, train=False, download=True, transform=transform_val)
        dataset_train = MyCIFAR100(args.data_path, train=True, download=True, transform=transform_train, 
                        noisy_labels=args.noisy_labels)
        dataset_val = datasets.CIFAR100(args.data_path, train=False, download=True, transform=transform_val)

    elif dataset == 'CIFAR10':
        dataset_train = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = datasets.CIFAR10(args.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'MNIST':
        dataset_train = MNIST_RGB(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = MNIST_RGB(args.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'FashionMNIST':
        dataset_train = FashionMNIST(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = FashionMNIST(args.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'SVHN':
        dataset_train = SVHN(args.data_path, split='train', download=True, transform=transform_train)
        dataset_val = SVHN(args.data_path, split='test', download=True, transform=transform_val)
    
    elif dataset == 'NotMNIST':
        dataset_train = NotMNIST(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = NotMNIST(args.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'Flower102':
        dataset_train = Flowers102(args.data_path, split='train', download=True, transform=transform_train)
        dataset_val = Flowers102(args.data_path, split='test', download=True, transform=transform_val)
    
    elif dataset == 'Cars196':
        dataset_train = StanfordCars(args.data_path, split='train', download=True, transform=transform_train)
        dataset_val = StanfordCars(args.data_path, split='test', download=True, transform=transform_val)
        
    elif dataset == 'CUB200':
        dataset_train = CUB200(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = CUB200(args.data_path, train=False, download=True, transform=transform_val).data
    
    elif dataset == 'Scene67':
        dataset_train = Scene67(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = Scene67(args.data_path, train=False, download=True, transform=transform_val).data

    elif dataset == 'TinyImagenet':
        dataset_train = TinyImagenet(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = TinyImagenet(args.data_path, train=False, download=True, transform=transform_val).data
        
    elif dataset == 'Imagenet-R':
        dataset_train = Imagenet_R(args.data_path, train=True, download=True, transform=transform_train,
                noisy_labels=args.noisy_labels).data
        dataset_val = Imagenet_R(args.data_path, train=False, download=True, transform=transform_val).data
    
    else:
        raise ValueError('Dataset {} not found.'.format(dataset))
    
    return dataset_train, dataset_val

def contaminate(dataset_train, shuffled_labels,classes_per_task,args):
    # get_transition_matrix
    transition_matrix = []
    for i in range(args.nb_classes):
        transition_matrix.append([])
        for j in range(args.nb_classes):
            transition_matrix[i].append(0.0)

    if args.in_task_noise:
        for _ in range(args.num_tasks):
            scope = shuffled_labels[:classes_per_task]
            shuffled_labels = shuffled_labels[classes_per_task:]

            if args.noisy_labels_type == "symmetry":
                for i in range(classes_per_task):
                    for j in range(classes_per_task):
                        if i == j:
                            transition_matrix[scope[i]][scope[j]] = 1.0 - args.noisy_labels_rate
                        else:
                            transition_matrix[scope[i]][scope[j]] = args.noisy_labels_rate / float(classes_per_task - 1)
                            
            elif args.noisy_labels_type == "pair":
                for i in range(classes_per_task):
                    transition_matrix[scope[i]][(scope[(i+1)%classes_per_task]) ] = args.noisy_labels_rate
                    for j in range(classes_per_task):
                        if i == j:
                            transition_matrix[scope[i]][scope[j]] = 1.0 - args.noisy_labels_rate
    else: # noise from out of task distribution, e.g., OOD
        if args.noisy_labels_type == "symmetry":
            for i in range(args.nb_classes):
                for j in range(args.nb_classes):
                    if i == j:
                        transition_matrix[i][j] = 1.0 - args.noisy_labels_rate
                    else:
                        transition_matrix[i][j] = args.noisy_labels_rate / float(args.nb_classes - 1)
        elif args.noisy_labels_type == "pair":
            for i in range(args.nb_classes):
                transition_matrix[i][(i + 1) % args.nb_classes] = args.noisy_labels_rate
                for j in range(args.nb_classes):
                    if i == j:
                        transition_matrix[i][j] = 1.0 - args.noisy_labels_rate

    # get_label_noise
    dataset_train.original_targets = deepcopy(dataset_train.targets)
    for trg_i, trg in enumerate(dataset_train.targets):
        noisy_label = np.random.choice(args.nb_classes, 1,\
                True, p=transition_matrix[trg])[0]
        dataset_train.targets[trg_i] = noisy_label


def split_single_dataset(dataset_train, dataset_val, args):
    nb_classes = len(dataset_val.classes)
    assert nb_classes % args.num_tasks == 0
    classes_per_task = nb_classes // args.num_tasks

    labels = [i for i in range(nb_classes)]
    
    split_datasets = list()
    mask = list()

    if args.shuffle:
        random.shuffle(labels)

    if args.noisy_labels:
        contaminate(dataset_train, shuffled_labels=labels,
                classes_per_task=classes_per_task,args=args)

    task_train_split_indices, task_test_split_indices = [], []
    for t_i in range(args.num_tasks):
        train_split_indices = []
        test_split_indices = []
        
        scope = labels[:classes_per_task]
        labels = labels[classes_per_task:]
        print("scope", scope)

        if args.sep_specialization and args.sep_criterion=='random': 
            # add randomized disjoint classes for prompts
            if t_i == 0:
                mask.append(scope)
                shuffle_labels = deepcopy(labels)
                random.shuffle(shuffle_labels)
                for _ in range(args.num_tasks-1): # 0-8
                    shuffle_scope = shuffle_labels[:classes_per_task]
                    shuffle_labels = shuffle_labels[classes_per_task:]
                    mask.append(shuffle_scope)
                print("mask", mask)
        elif args.sep_specialization and args.sep_criterion=='random_1T': 
            # add randomized disjoint classes for prompts
            if t_i == 0:
                shuffle_labels = deepcopy([i for i in range(nb_classes)])
                random.shuffle(shuffle_labels)
                for _ in range(args.num_tasks): # 0-9
                    shuffle_scope = shuffle_labels[:classes_per_task]
                    shuffle_labels = shuffle_labels[classes_per_task:]
                    mask.append(shuffle_scope)
                print("mask", mask)
        else:
            mask.append(scope)

        for k in range(len(dataset_train.targets)):
            if int(dataset_train.targets[k]) in scope:
                train_split_indices.append(k)
                
        for h in range(len(dataset_val.targets)):
            if int(dataset_val.targets[h]) in scope:
                test_split_indices.append(h)
        
        if not args.blurryCL:
            # print(train_split_indices[:10])
            print('#tr, te: ', len(train_split_indices), len(test_split_indices))
            subset_train, subset_val =  Subset(dataset_train, train_split_indices), Subset(dataset_val, test_split_indices)
            split_datasets.append([subset_train, subset_val])

        task_train_split_indices += [train_split_indices]
        task_test_split_indices += [test_split_indices]
        
    
    if args.blurryCL:
        task_trainM, task_trainN = [], []
        for t in task_train_split_indices: # t is list of indices
            sub_task_trainN = []
            # t = np.array(t)
            taskM = list(np.random.choice(t, int(len(t)*args.blurryM), replace=False))
            taskN = list(set(t) - set(taskM))
            taskN_size = len(taskN)

            task_trainM.append(taskM)
            for _ in range(len(task_train_split_indices)-1):
                sub_task_trainN.append(list(np.random.choice(taskN, taskN_size//(len(task_train_split_indices)-1))))
            task_trainN.append(sub_task_trainN)

        task_train_blurry_indices = []
        for task_idx, task_inds in enumerate(task_trainM):
            other_task_inds = []
            for j in range(len(task_train_split_indices)):
                if j != task_idx:
                    other_task_inds = other_task_inds + task_trainN[j].pop(0)
            blurry_task_inds = task_inds + other_task_inds
            task_train_blurry_indices.append(blurry_task_inds)
            
            print(blurry_task_inds[:10])
            subset_train =  Subset(dataset_train, blurry_task_inds)
            subset_val = Subset(dataset_val, task_test_split_indices[task_idx])
            split_datasets.append([subset_train, subset_val])
            print('#tr, te: ', len(blurry_task_inds), len(task_test_split_indices[task_idx]))

    return split_datasets, mask

def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        scale = (0.08, 1.0)
        ratio = (3. / 4., 4. / 3.)
        transform = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    
    return transforms.Compose(t)


    