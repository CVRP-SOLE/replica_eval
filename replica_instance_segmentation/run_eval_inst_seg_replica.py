import os
import numpy as np
import clip
import torch
import pyviz3d.visualizer as viz
import open3d as o3d
import matplotlib.pyplot as plt
import pdb
from eval_semantic_instance_orig_replica import evaluate
from replica_constants import CLASS_LABELS_REPLICA, VALID_CLASS_IDS_REPLICA, COLOR_MAP_REPLICA
import tqdm


class InstSegEvaluator():
    def __init__(self, dataset_type, clip_model_type, sentence_structure):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset_type = dataset_type
        self.clip_model_type = clip_model_type
        self.clip_model = self.get_clip_model(clip_model_type)
        self.query_sentences = self.get_query_sentences(dataset_type, sentence_structure)
        self.feature_size = self.get_feature_size(clip_model_type)
        self.text_query_embeddings = self.get_text_query_embeddings().numpy() #torch.Size([20, 768])
        self.set_label_and_color_mapper(dataset_type)

    def get_query_sentences(self, dataset_type, sentence_structure="a {} in a scene"):
        if dataset_type == 'replica':
            label_list = list(CLASS_LABELS_REPLICA)
        else:
            raise NotImplementedError
        return [sentence_structure.format(label) for label in label_list]

    def get_clip_model(self, clip_model_type):
        clip_model, _ = clip.load(clip_model_type, self.device)
        return clip_model

    def get_feature_size(self, clip_model_type):
        if clip_model_type == 'ViT-L/14' or clip_model_type == 'ViT-L/14@336px':
            return 768
        elif clip_model_type == 'ViT-B/32':
            return 512
        else:
            raise NotImplementedError

    def get_text_query_embeddings(self):
        # ViT_L14_336px for OpenSeg, clip_model_vit_B32 for LSeg - ViT_L14 was used in our ConceptFusion implementation
        text_query_embeddings = torch.zeros((len(self.query_sentences), self.feature_size))

        for label_idx, sentence in enumerate(self.query_sentences):
            #print(label_idx, sentence) #CLASS_LABELS_20[label_idx],
            text_input_processed = clip.tokenize(sentence).to(self.device)
            with torch.no_grad():
                sentence_embedding = self.clip_model.encode_text(text_input_processed)

            sentence_embedding_normalized =  (sentence_embedding/sentence_embedding.norm(dim=-1, keepdim=True)).float().cpu()
            text_query_embeddings[label_idx, :] = sentence_embedding_normalized

        return text_query_embeddings
    
    def set_label_and_color_mapper(self, dataset_type):
        if dataset_type == 'replica':
            self.label_mapper = np.vectorize({idx: el for idx, el in enumerate(VALID_CLASS_IDS_REPLICA)}.get)
            self.id2label = np.vectorize({el: idx for idx, el in enumerate(VALID_CLASS_IDS_REPLICA)}.get)
            self.color_mapper = np.vectorize(COLOR_MAP_REPLICA.get)
        else:
            raise NotImplementedError

    def compute_classes_per_mask(self, masks_path, mask_features_path, keep_first=None):
        masks = torch.load(masks_path)
        mask_features = np.load(mask_features_path)

        if keep_first is not None:
            masks = masks[:, 0:keep_first]
            mask_features = mask_features[0:keep_first, :]

        # normalize mask features
        mask_features_normalized = mask_features/np.linalg.norm(mask_features, axis=1)[..., None]

        similarity_scores = mask_features_normalized@self.text_query_embeddings.T #(177, 20)
        max_class_similarity_scores = np.max(similarity_scores, axis=1) # does not correspond to class probabilities
        max_ind = np.argmax(similarity_scores, axis=1)
        max_ind_remapped = self.label_mapper(max_ind)
        pred_classes = max_ind_remapped

        return masks, pred_classes, max_class_similarity_scores
    

    def compute_classes_per_mask_diff_scores(self, masks_path, mask_features_path, score_data, keep_first=None):
        pred_masks = torch.load(masks_path)
        mask_features = np.load(mask_features_path)

        keep_mask = np.asarray([True for el in range(pred_masks.shape[1])])
        if keep_first:
            keep_mask[keep_first:] = False
        #num_point_ratio_mask = np.round(pred_masks.sum(axis=0)/pred_masks.shape[0], 3)*100 > 10
        #keep_mask[num_point_ratio_mask] = False

        # normalize mask features
        #pdb.set_trace()
        mask_features_normalized = mask_features/np.linalg.norm(mask_features, axis=1)[..., None]
        mask_features_normalized[np.isnan(mask_features_normalized) | np.isinf(mask_features_normalized)] = 0.0

        per_class_similarity_scores = mask_features_normalized@self.text_query_embeddings.T #(177, 20)
        max_class_similarity_scores = np.max(per_class_similarity_scores, axis=1) # does not correspond to class probabilities
        max_ind = np.argmax(per_class_similarity_scores, axis=1)
        max_ind_remapped = self.label_mapper(max_ind)

        pred_masks = pred_masks[:, keep_mask]
        pred_classes = max_ind_remapped[keep_mask]

        if score_data['score_type'] == 'all_ones':
            pred_scores = np.ones(pred_classes.shape)

        elif score_data['score_type'] == 'orig_mask_scores':
            pred_scores = torch.load(score_data['scores_path'])
            pred_scores = pred_scores[keep_mask]

        elif score_data['score_type'] == 'similarity_scores':
            pred_scores = max_class_similarity_scores
            pred_scores = pred_scores[keep_mask]

        elif score_data['score_type'] == 'heatmap_scores':
            mask_pred =  torch.load(masks_path)
            heatmap_pred = torch.load(score_data['heatmap_path'])
            result_pred_mask = (mask_pred > 0).astype(float)
            pred_scores = (heatmap_pred * result_pred_mask).sum(0) / (result_pred_mask.sum(0) + 1e-6) 
            pred_scores = pred_scores[keep_mask]

        elif score_data['score_type'] == 'heatmap_times_similarity_score' or score_data['score_type'] == 'heatmap_plus_similarity_score':
            mask_pred =  torch.load(masks_path)
            heatmap_pred = torch.load(score_data['heatmap_path'])
            result_pred_mask = (mask_pred > 0).astype(float)
            heatmap_based_scores = (heatmap_pred * result_pred_mask).sum(0) / (result_pred_mask.sum(0) + 1e-6) 
            heatmap_based_scores = heatmap_based_scores[keep_mask]

            # rescale similarity based scores
            similarity_based_scores = max_class_similarity_scores[keep_mask]
            similarity_based_scores = (similarity_based_scores-similarity_based_scores.min())/(similarity_based_scores.max()-similarity_based_scores.min())
            
            #pred_scores =  heatmap_based_scores * score_data['score_type']
            if score_data['score_type'] == 'heatmap_times_similarity_score':
                pred_scores =  heatmap_based_scores * similarity_based_scores
            elif score_data['score_type'] == 'heatmap_plus_similarity_score':
                pred_scores =  (heatmap_based_scores * score_data['heatmap_score_weight']) + similarity_based_scores 
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError
     
        #print(pred_masks.shape, pred_scores.shape, pred_classes.shape)#pred_masks.shape, pred_scores.shape, pred_classes.shape #((237360, 41), (41,), (41,))
    
        return pred_masks, pred_classes, pred_scores


    def evaluate(self, pred_masks, pred_classes, pred_scores, scene_gt_dir, dataset, output_file='temp_output.txt'):
        #pred_masks.shape, pred_scores.shape, pred_classes.shape #((237360, 177), (177,), (177,))
        preds = {}
        preds['scene0011_00'] = {
            'pred_masks': pred_masks,
            'pred_scores': pred_scores,
            'pred_classes': pred_classes}

        inst_AP = evaluate(preds, scene_gt_dir, output_file=output_file, dataset=dataset)
        # read .txt file: scene0000_01.txt has three parameters each row: the mask file for the instance, the id of the instance, and the score. 

        return inst_AP

    def evaluate_full(self, preds, scene_gt_dir, dataset, output_file='temp_output.txt'):
        #pred_masks.shape, pred_scores.shape, pred_classes.shape #((237360, 177), (177,), (177,))

        inst_AP = evaluate(preds, scene_gt_dir, output_file=output_file, dataset=dataset)
        # read .txt file: scene0000_01.txt has three parameters each row: the mask file for the instance, the id of the instance, and the score. 

        return inst_AP


    def visualize_embeddings_given_a_query(self, scene_path, masks_path, mask_features_path, query_text, viz_save_dir, cmap_name='turbo', point_size=35, keep_first=None):
        cm = plt.get_cmap(cmap_name)
        def get_query_color(similarity, max_sim=1.0):
            return (np.asarray(cm(similarity/max_sim)[0:3])*255).astype(int)

        # load and normalize mask features, load scene pcd and masks
        scene = o3d.io.read_point_cloud(scene_path)
        masks = torch.load(masks_path)
        mask_features = np.load(mask_features_path)

        if keep_first is not None:
            masks = masks[:, 0:keep_first]
            mask_features = mask_features[0:keep_first, :]

        #pdb.set_trace()
        mask_features_normalized = mask_features/np.linalg.norm(mask_features, axis=1)[..., None]
        invalid_mask = np.isnan(mask_features_normalized.sum(axis=1)) | np.isinf(mask_features_normalized.sum(axis=1))
        mask_features_normalized[np.isnan(mask_features_normalized) | np.isinf(mask_features_normalized)] = 0.0

        text_input_processed = clip.tokenize(query_text).to(self.device)
        with torch.no_grad():
            text_embedding = self.clip_model.encode_text(text_input_processed)
        text_embedding_normalized =  (text_embedding/text_embedding.norm(dim=-1, keepdim=True)).float().cpu().numpy()
        
        per_mask_scores = np.squeeze((mask_features_normalized@text_embedding_normalized.T))

        print('Setting up visualizer...')
        # we set up a visualizer
        v = viz.Visualizer()

        # load scene pcd and masks
        scene.estimate_normals()
        point_positions = np.asarray(scene.points) - np.mean(np.asarray(scene.points), axis=0)
        point_colors = np.asarray(scene.colors)
        point_normals = np.asarray(scene.normals)

        v.add_points('RGB Color', point_positions, point_colors, point_normals, point_size=point_size, visible=False)

        max_sim = per_mask_scores.max()
        for mask_id in range(masks.shape[1]):
            if invalid_mask[mask_id]:
                continue
            mask_point_ids = np.argwhere(masks[:, mask_id])[:, 0]
            mask_point_positions = point_positions[mask_point_ids, :]
            mask_semantic_colors = np.asarray([get_query_color(per_mask_scores[mask_id], max_sim) for el in range(len(mask_point_positions))])
            mask_point_normals = point_normals[mask_point_ids, :]
            v.add_points('mask_'+str(mask_id).zfill(3), mask_point_positions, mask_semantic_colors, mask_point_normals, point_size=point_size, visible=True)
        v.save(viz_save_dir)

def test_pipeline_full_replica(per_mask_features_dir,
                         gt_dir = '/home/ayca/openmask/openmask/eval/replica_gt/instances',
                         pred_root_dir = '/media/ayca/Elements/ayca/OpenMask3D/replica_class_agn_masks',
                         dataset_type = 'replica',
                         clip_model_type = 'ViT-L/14@336px',
                         sentence_structure = "a {} in a scene",
                         score_type='all_ones',
                         keep_first = None,
                         feature_file_template='',
                         masks_template='_mesh_masks.pt'):


    evaluator = InstSegEvaluator(dataset_type, clip_model_type, sentence_structure)

    score_data = {'score_type': score_type}
    print('[INFO]:', dataset_type, clip_model_type, sentence_structure, score_type)


    scene_names = ['office0', 'office1','office2', 'office3', 'office4', 'room0', 'room1', 'room2']
    
    preds = {}

    for scene_name in tqdm.tqdm(scene_names[:]):
        masks_path = os.path.join(pred_root_dir, scene_name + masks_template)
        scene_per_mask_feature_path = os.path.join(per_mask_features_dir, feature_file_template.format(scene_name))

        if not os.path.exists(scene_per_mask_feature_path):
            print('--- SKIPPING ---', scene_per_mask_feature_path)
            continue

        assert score_type=='all_ones'
        score_data = {'score_type': score_type}
        
        pred_masks, pred_classes, pred_scores = evaluator.compute_classes_per_mask_diff_scores(masks_path=masks_path, 
                                                                                               mask_features_path=scene_per_mask_feature_path,
                                                                                               score_data=score_data, 
                                                                                               keep_first=keep_first)
        
        preds[scene_name] = {
            'pred_masks': pred_masks,
            'pred_scores': pred_scores,
            'pred_classes': pred_classes}

    inst_AP = evaluator.evaluate_full(preds, gt_dir, dataset=dataset_type)

if __name__ == '__main__':

    
    # pipeline 5, replica, all_ones, "a {} in a scene"
    sentence_structure = "a {} in a scene" #"picture of a {}" #"photo of a {}" #"a {} in a scene" #"{}" #
    test_pipeline_full_replica(per_mask_features_dir = '/PATH/TO/PER/MASK/OPENMASK3D/FEATURES/FOR/REPLICA',
                         gt_dir = 'replica_gt/instances',
                         pred_root_dir = '/PATH/TO/PREDICTED/MASKS/FOR/REPLICA/',
                         dataset_type = 'replica',
                         clip_model_type = 'ViT-L/14@336px',
                         sentence_structure = sentence_structure,
                         score_type='all_ones',
                         keep_first = None,
                         feature_file_template='{}_openmask3d_features.npy')
