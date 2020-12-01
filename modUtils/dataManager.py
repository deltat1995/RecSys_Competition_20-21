import numpy as np
import pandas as pd
import scipy.sparse as sps

from sklearn.model_selection import train_test_split


def load_data() -> pd.DataFrame:
    """CARICA I DATI DELLA URM DELLA COMPETITION"""

    return pd.read_csv("./data/data_train.csv",
                       sep=",",
                       names=["user_id", "item_id", "impl_rating"],
                       header=None,
                       skiprows=1,
                       dtype={"user_id": np.int32,
                              "item_id": np.int32,
                              "impl_rating": np.int32})


def load_icm() -> pd.DataFrame:
    """CARICA I DATI DELLA ICM DELLA COMPETITION"""

    return pd.read_csv("./data/data_ICM_title_abstract.csv",
                       sep=",",
                       names=["item_id", "feature_id", "weighted_value"],
                       header=None,
                       skiprows=1,
                       dtype={"item_id": np.int32,
                              "feature_id": np.int32,
                              "weighted_value": np.float})


def preprocess_data(ratings: pd.DataFrame) -> pd.DataFrame:
    """ESTENDE LA URM MATRIX CON LA MAPPATURA DI USER E ITEM IDS"""
    unique_users = ratings.user_id.unique()
    unique_items = ratings.item_id.unique()

    num_users, min_user_id, max_user_id = unique_users.size, unique_users.min(), unique_users.max()
    num_items, min_item_id, max_item_id = unique_items.size, unique_items.min(), unique_items.max()

    print(f'Numero di users: {num_users}, UserId minimo: {min_user_id}, UserId massimo: {max_user_id}')
    print(f'Numero di items: {num_items}, ItemId minimo: {min_item_id}, ItemId massimo: {max_item_id}')
    print("Sparsity della URM: %.3f %%" % (len(ratings) * 100 / (num_users * num_items)))

    mapping_user_id = pd.DataFrame({"mapped_user_id": np.arange(num_users), "user_id": unique_users})
    mapping_item_id = pd.DataFrame({"mapped_item_id": np.arange(num_items), "item_id": unique_items})

    ratings = pd.merge(left=ratings,
                       right=mapping_user_id,
                       how="inner",
                       on="user_id")

    ratings = pd.merge(left=ratings,
                       right=mapping_item_id,
                       how="inner",
                       on="item_id")
    return ratings


def preprocess_icm(icm_matrix: pd.DataFrame) -> pd.DataFrame:
    """ESTENDE LA ICM MATRIX CON LA MAPPATURA DI ITEM E FEATURE IDS"""
    unique_items = icm_matrix.item_id.unique()
    unique_features = icm_matrix.feature_id.unique()

    num_items, min_item_id, max_item_id = unique_items.size, unique_items.min(), unique_items.max()
    num_features, min_feature_id, max_feature_id = unique_features.size, \
                                                   unique_features.min(), unique_features.max()

    print(f'Numero di items: {num_items}, ItemId minimo: {min_item_id}, ItemId massimo: {max_item_id}')
    print(
        f'Numero di features: {num_features}, FeatureId minimo: {min_feature_id}, FeatureId massimo: {max_feature_id}')
    print("Sparsity della ICM: %.3f %%" % (len(icm_matrix) * 100 / (num_items * num_features)))

    mapping_item_id = pd.DataFrame({"mapped_item_id": np.arange(num_items), "item_id": unique_items})
    mapping_feature_id = pd.DataFrame({"mapped_feature_id": np.arange(num_features), "feature_id": unique_features})

    icm_matrix = pd.merge(left=icm_matrix,
                          right=mapping_item_id,
                          how="inner",
                          on="item_id")

    icm_matrix = pd.merge(left=icm_matrix,
                          right=mapping_feature_id,
                          how="inner",
                          on="feature_id")
    return icm_matrix


def dataset_splits(ratings, num_users, num_items, val_perc: float, test_perc: float) -> \
        (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """DIVIDE IL DATASET DEL URM IN TRAINING, VALUATION AND TEST SETS"""
    seed = 9876

    (uid_training, uid_test,
     iid_training, iid_test,
     ratings_training, ratings_test) = train_test_split(ratings.mapped_user_id,
                                                        ratings.mapped_item_id,
                                                        ratings.impl_rating,
                                                        test_size=test_perc,
                                                        shuffle=True,
                                                        random_state=seed)
    (uid_training, uid_validation,
     iid_training, iid_validation,
     ratings_training, ratings_validation) = train_test_split(uid_training,
                                                              iid_training,
                                                              ratings_training,
                                                              test_size=val_perc)

    urm_train = sps.csr_matrix((ratings_training, (uid_training, iid_training)), shape=(num_users, num_items))
    urm_val = sps.csr_matrix((ratings_validation, (uid_validation, iid_validation)), shape=(num_users, num_items))
    urm_test = sps.csr_matrix((ratings_test, (uid_test, iid_test)), shape=(num_users, num_items))

    return urm_train, urm_val, urm_test


def prepare_submission(ratings: pd.DataFrame, urm_train: sps.csr_matrix, recommender: object) -> list:
    """CREA LA LISTA DI TUPLE (USERID,LISTA DI ITEMIDS RACCOMANDATI)"""
    users_to_recommend = pd.read_csv("./data/data_target_users_test.csv",
                                     names=["user_id"],
                                     header=None,
                                     skiprows=1,
                                     dtype={"user_id": np.int32})

    user_ids_and_mappings = ratings[ratings.user_id.isin(users_to_recommend.user_id)][
        ["user_id", "mapped_user_id"]].drop_duplicates().sort_values(by=['user_id'])

    mapping_to_item_id = dict(zip(ratings.mapped_item_id, ratings.item_id))

    recommendation_length = 10
    submission = []
    for idx, row in user_ids_and_mappings.iterrows():
        user_id = row.user_id
        mapped_user_id = row.mapped_user_id

        recommendations = recommender.recommend(user_id=mapped_user_id,
                                                urm_train=urm_train,
                                                at=recommendation_length)

        submission.append((user_id, [mapping_to_item_id[item_id] for item_id in recommendations]))

    return submission


def write_submission(submission, namefile: str) -> None:
    """CREA UN FILE COL NOME 'namefile' RISPETTANDO IL FORMATO DI KAGGLE"""
    with open("./submissions/" + namefile + ".csv", "w") as f:
        f.write("user_id,item_list\n")
        for user_id, items in submission:
            f.write(f"{user_id},{' '.join([str(item) for item in items])}\n")
