import sys
import pandas as pd
import numpy as np
import copy
from datetime import datetime
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler, MultiLabelBinarizer
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, concatenate, TimeDistributed, Dropout, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.losses import BinaryFocalCrossentropy

# print(tf.config.list_physical_devices('GPU'))

index_df = pd.read_csv("course_recommender_course_index.csv")
index_df["course_level"] = index_df["Course_Number"].str[0].astype(int)
index_df["course_index"] = index_df["course_index"].astype(int)

core_df = pd.read_csv("course_codes_with_core_curriculum_areas_flagged.csv")

## make a core_matrix and grab column names for later reference
core_matrix = core_df.set_index("Course_ID")
core_columns = [col for col in core_matrix.columns]

## make a way to map courses to indexes and vice-versa
course_to_idx = dict(zip(index_df['Course_ID'], index_df['course_index']))
idx_to_course = {v: k for k, v in course_to_idx.items()}

index_df['course_index'] = pd.to_numeric(index_df['course_index']) 
course_to_idx = dict(zip(index_df['Course_ID'], index_df['course_index']))
idx_to_course = {v: k for k, v in course_to_idx.items()}

## calculate VOCAB_SIZE based on the maximum index
VOCAB_SIZE = len(index_df) + 1

subj_vocab_size = len(index_df["Subject_Code"].unique())

## find max number of terms, max number of courses, number of unique majors and number of unique class standings
max_sizes = pd.read_csv("course_recommender_input_stats.csv")
max_terms = int(max_sizes.loc[0,"max_terms"])
max_courses_per_term = int(max_sizes.loc[0, "max_courses"])
major_vocab_size = int(max_sizes.loc[0, "num_major_values"])
standing_vocab_size = int(max_sizes.loc[0, "num_standing_values"])


## prepare to engineer features
course_features_df = index_df[['Course_ID', 'Department_Desc']].copy()
course_features_df['course_level'] = pd.to_numeric(index_df['Course_Number'].str.slice(0, 4))

subject_encoder = LabelEncoder().fit(index_df["Subject_Code"])
course_level_encoder = LabelEncoder().fit(index_df["course_level"])

course_features_df = StandardScaler().fit_transform(course_features_df[['course_level']])

level_vocab_size = len(course_level_encoder.classes_)

# --- Binarize Target Variable ---
all_course_indices = index_df['course_index'].unique().tolist()

data_to_fit = [[index] for index in all_course_indices]
mlb = MultiLabelBinarizer()
mlb.fit(data_to_fit)

## set up some functions

# Function to get core vector for a course
def get_core_vector(course_id):
    return list(core_matrix.loc[course_id, core_columns])

def get_subj(course_id):
    ## the subject is the first four characters of the Course_ID
    return course_id[0:4]

def get_course_level(course_id):
    ## the level is the fifth character of the Course_ID
    return int(course_id[4])

# def create_full_feature_sequences(main_df, student_features_df):
#     """
#     Creates structured sequences where each student's history is a list of term-course-lists.
#     Also creates the corresponding next-term target and gathers all features.
#     """
#     sequences = {
#         'student_id': [], 'history': [], 'course_subjs': [], 'student_major': [], 'student_standing': [],
#         'student_gpa': [], 'target_courses': [], 'history_core': [], 'course_levels': []
#     }
    
#     student_groups = main_df.sort_values(['ID', 'Term_Code']).groupby('ID')
    
#     ## loop through students
#     for student_id, group in student_groups:
#         term_groups = group.groupby('Term_Code')['Course_ID'].apply(list)
#         term_indices = [[course_to_idx.get(course) for course in term] for term in term_groups]

#         ## if a student was only here for one term since 201801, don't include them.
#         if len(term_indices) < 2:
#             continue

#         term_core_vectors = [[get_core_vector(course) for course in term] for term in term_groups]
#         term_subj_vectors = [subject_encoder.transform([get_subj(course) for course in term]) for term in term_groups]
#         term_course_level_vectors = [course_level_encoder.transform([get_course_level(course) for course in term]) for term in term_groups]

            
#         student_terms = student_features_df[student_features_df['ID'] == student_id]    ## multiple rows per student (1 per term)
            


#         ## loop through the student's terms
#         for i in range(1, len(term_indices)):
#             history = term_indices[:i]
#             target = term_indices[i]
        
#             major = student_terms['major_encoded'].iloc[i - 1]
#             standing = student_terms['standing_encoded'].iloc[i - 1]
#             gpa = student_terms['standardized_gpa'].iloc[i - 1]

#             core_history = term_core_vectors[:i]
#             subj_history = term_subj_vectors[:i]
#             course_level_history = term_course_level_vectors[:i]
            
#             sequences['history_core'].append(core_history)
#             sequences["course_subjs"].append(subj_history)
#             sequences['student_id'].append(student_id)
#             sequences['history'].append(history)
#             sequences['target_courses'].append(target)
#             sequences['student_major'].append(major)
#             sequences['student_standing'].append(standing)
#             sequences['student_gpa'].append(gpa)
#             sequences['course_levels'].append(course_level_history)
            
#     return sequences

# def pad_3d(sequences):
#     padded = np.zeros((len(sequences), max_terms, max_courses_per_term), dtype='int32')
#     for i, hist in enumerate(sequences):
#         for j, term in enumerate(hist):
#             padded[i, j, :len(term)] = term[:max_courses_per_term]
#     return padded
# def pad_core_4d(history_core, max_terms, max_courses, num_core_features):
#     padded = np.zeros((len(history_core), max_terms, max_courses, num_core_features))
#     for i, student_hist in enumerate(history_core):
#         for j, term in enumerate(student_hist):
#             for k, core_vec in enumerate(term):
#                 if k < max_courses:
#                     padded[i, j, k, :] = core_vec
#     return padded






# --- Define Model Inputs ---
input_history = Input(shape=(max_terms, max_courses_per_term), name='history_input')
input_subject = Input(shape=(max_terms, max_courses_per_term), name='subject_input')
input_course_level = Input(shape=(max_terms, max_courses_per_term), name='course_level_input')
input_major = Input(shape=(1,), name='major_input')
input_standing = Input(shape=(1,), name='standing_input')
input_gpa = Input(shape=(1,), name='gpa_input')
# input_history_core = Input(shape=(max_terms, max_courses_per_term, len(core_columns)), name='history_core_input')


# --- Path 1: Process Course History ---
course_embedding_layer = Embedding(input_dim=VOCAB_SIZE, output_dim=64, mask_zero=True)
subject_embedding_layer = Embedding(input_dim=subj_vocab_size, output_dim=32, mask_zero=True)
level_embedding_layer = Embedding(input_dim=level_vocab_size, output_dim=8, mask_zero=True)


# The TimeDistributed layer applies the embedding to each term's list of courses.
embedded_courses = TimeDistributed(course_embedding_layer)(input_history)
embedded_subjects = TimeDistributed(subject_embedding_layer)(input_subject)
embedded_levels = TimeDistributed(level_embedding_layer)(input_course_level)
combined_course_features = concatenate([
    embedded_courses, 
    # input_history_core, 
    embedded_subjects, 
    embedded_levels
])

# --- Use GlobalAveragePooling1D to average the embeddings of all courses within each term ---
# This creates a single, stable vector representation for each term
#term_vectors = TimeDistributed(GlobalAveragePooling1D())(embedded_courses)
term_vectors = TimeDistributed(GlobalAveragePooling1D())(combined_course_features)


# The LSTM now processes this improved sequence of term vectors.
lstm_out = LSTM(128, dropout=0.2, recurrent_dropout=0.2)(term_vectors)

# --- Path 2: Process Student Static Features ---
major_embedding = Embedding(input_dim=major_vocab_size, output_dim=10)(input_major)
standing_embedding = Embedding(input_dim=standing_vocab_size, output_dim=5)(input_standing)
major_flat = tf.keras.layers.Flatten()(major_embedding)
standing_flat = tf.keras.layers.Flatten()(standing_embedding)
static_features_concat = concatenate([major_flat, standing_flat, input_gpa])


# --- Combine All Paths ---
final_concat = concatenate([lstm_out, static_features_concat])
dense_1 = Dropout(0.5)(Dense(256, activation='relu')(final_concat))
dense_2 = Dropout(0.4)(Dense(128, activation='relu')(dense_1))
output_layer = Dense(len(mlb.classes_), activation='sigmoid', name='output')(dense_2)

# --- Create and Compile Model ---
full_feature_model = Model(
    inputs=[
        input_history, 
        # input_history_core, 
        input_subject,
        input_major, 
        input_standing, 
        input_gpa, 
        input_course_level 
    ],
    outputs=output_layer
)

full_feature_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=BinaryFocalCrossentropy(),
    metrics=[Precision(name='precision'), Recall(name='recall'), AUC(name='auc')]
)

# print(full_feature_model.summary())




# grab the testing data
# test_data = pd.read_csv("course_recommender_full_test_data.csv")

# test_data["is_fall"] = test_data["Term_Code"] % 10 == 8
# test_data["is_spring"] = test_data["Term_Code"] % 10 == 2
# test_data["is_summer"] = test_data["Term_Code"] % 10 == 5

# test_data['MATRICULATION_TERM'] = pd.to_numeric(test_data['MATRICULATION_TERM'])
# test_data['Term_Code'] = pd.to_numeric(test_data['Term_Code'])
# test_data['Hours_Attempted'] = pd.to_numeric(test_data['Hours_Attempted'])
# test_data['Hours_Passed'] = pd.to_numeric(test_data['Hours_Passed'])

# test_data = pd.merge(test_data, core_df, how='left', on='Course_ID')

# test_student_performance = test_data.groupby(['ID', 'Term_Code']).agg(
#     major_encoded=('major_encoded', 'max'),
#     standing_encoded=('standing_encoded', 'max'),
#     standardized_gpa = ('standardized_gpa', 'max')               ## this is a standardized gpa for just this term
# ).reset_index()


# test_student_features_df = test_student_performance[["ID", "Term_Code", "major_encoded", "standing_encoded", "standardized_gpa"]]

# test_sequences = create_full_feature_sequences(test_data, test_student_features_df)

# X_test_history_core = pad_core_4d(test_sequences['history_core'], max_terms, max_courses_per_term, len(core_columns))
# X_test_hist = pad_3d(test_sequences['history'])
# X_test_subj = pad_3d(test_sequences['course_subjs'])
# X_test_course_level = pad_3d(test_sequences['course_levels'])
# X_test_major = np.array(test_sequences['student_major'])
# X_test_standing = np.array(test_sequences['student_standing'])
# X_test_gpa = np.array(pd.DataFrame(np.array(test_sequences['student_gpa']).reshape(-1, 1), columns=["gpa"]))

# X_test = {
#     'history_input': X_test_hist,
#     'history_core_input': X_test_history_core,
#     'major_input': X_test_major,
#     'standing_input': X_test_standing,
#     'gpa_input': X_test_gpa,
#     'subject_input': X_test_subj,
#     'course_level_input': X_test_course_level
# }

# y_test = mlb.transform(test_sequences['target_courses'])


print("starting x_test file")

# np.savez_compressed('x_test_file.npz', **X_test)

loaded_X_test = np.load("x_test_file.npz")
X_test = {
    'history_input':loaded_X_test['history_input'],
    'history_core_input':loaded_X_test['history_core_input'],
    'major_input':loaded_X_test['major_input'],
    'standing_input':loaded_X_test['standing_input'],
    'gpa_input':loaded_X_test['gpa_input'],
    'subject_input':loaded_X_test['subject_input'],
    'course_level_input':loaded_X_test['course_level_input']
}


print("starting y_test file")

# y_test_prepped = {"y_test": y_test}
# np.savez_compressed('y_test_file.npz', **y_test_prepped)

loaded_y_test = np.load('y_test_file.npz')

y_test = loaded_y_test["y_test"]







print("Starting training data")

## grab the correct batch of training data and prepare it
# training_data = pd.read_csv("course_recommender_full_training_data.csv")

# training_data["is_fall"] = training_data["Term_Code"] % 10 == 8
# training_data["is_spring"] = training_data["Term_Code"] % 10 == 2
# training_data["is_summer"] = training_data["Term_Code"] % 10 == 5

# training_data['MATRICULATION_TERM'] = pd.to_numeric(training_data['MATRICULATION_TERM'])
# training_data['Term_Code'] = pd.to_numeric(training_data['Term_Code'])
# training_data['Hours_Attempted'] = pd.to_numeric(training_data['Hours_Attempted'])
# training_data['Hours_Passed'] = pd.to_numeric(training_data['Hours_Passed'])

# training_data = pd.merge(training_data, core_df, how='left', on='Course_ID')

# training_student_performance = training_data.groupby(['ID', 'Term_Code']).agg(
#     major_encoded=('major_encoded', 'max'),
#     standing_encoded=('standing_encoded', 'max'),
#     standardized_gpa = ('standardized_gpa', 'max')           ## this is a standardized gpa for just this term
# ).reset_index()


# training_student_features_df = training_student_performance[["ID", "Term_Code", "major_encoded", "standing_encoded", "standardized_gpa"]]

# train_sequences = create_full_feature_sequences(training_data, training_student_features_df)

# X_train_history_core = pad_core_4d(train_sequences['history_core'], max_terms, max_courses_per_term, len(core_columns))
# X_train_hist = pad_3d(train_sequences['history'])
# X_train_subj = pad_3d(train_sequences['course_subjs'])
# X_train_course_level = pad_3d(train_sequences['course_levels'])
# X_train_major = np.array(train_sequences['student_major'])
# X_train_standing = np.array(train_sequences['student_standing'])
# X_train_gpa = np.array(pd.DataFrame(np.array(train_sequences['student_gpa']).reshape(-1, 1), columns=["gpa"]))

# X_train = {
#     'history_input': X_train_hist,
#     'history_core_input': X_train_history_core,
#     'major_input': X_train_major,
#     'standing_input': X_train_standing,
#     'gpa_input': X_train_gpa,
#     'subject_input': X_train_subj,
#     'course_level_input': X_train_course_level
# }

# y_train = mlb.transform(train_sequences['target_courses'])





print("starting x_train file")

# np.savez_compressed('x_train_file.npz', **X_train)

loaded_x_train = np.load('x_train_file.npz')

X_train = {
    'history_input': loaded_x_train['history_input'],
    'history_core_input': loaded_x_train['history_core_input'],
    'major_input': loaded_x_train['major_input'],
    'standing_input': loaded_x_train['standing_input'],
    'gpa_input': loaded_x_train['gpa_input'],
    'subject_input': loaded_x_train['subject_input'],
    'course_level_input': loaded_x_train['course_level_input']
}



print("starting y_train file")

# y_train_prepped = {"y_train": y_train}
# np.savez_compressed('y_train_file.npz', **y_train_prepped)

loaded_y_train = np.load('y_train_file.npz')

y_train = loaded_y_train["y_train"]





## grab actuals for each student-instance (for use later in model evaluation)
actuals = [
    np.where(binary_course_set == 1)[0] + 1            ## +1 because courses are 1-indexed, while probability indexes start at 0
    for binary_course_set in y_test
]






## train the model
print(f"--- Starting Model Training  ---")
history = full_feature_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=40,
    batch_size=32,
    callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)],
    verbose=1
)









print(f"--- evaluating model ---")

## do preliminary evaluation of model

## grab top 10 test predictions for each student-instance
test_prediction = full_feature_model.predict(X_test)
top_tens = [
    np.argsort(probability_set)[-10:][::-1] + 1       ## +1 because courses are 1-indexed, while probability indexes start at 0
    for probability_set in test_prediction
]


## copy X_test and add some things
numbered_X_test = copy.deepcopy(X_test)
numbered_X_test["sample_number"] = [j for j in range(len(y_test))]
numbered_X_test["actual"] = actuals
numbered_X_test["predicted"] = top_tens

## pivot the dictionary so that each key in the new dictionary represents a student-term instance
pivoted_dictionary = {
    sample_number: {
        'current_term_actual': actual,
        'current_term_prediction': predicted
    }
    for sample_number, actual, predicted in zip(
        numbered_X_test["sample_number"],
        numbered_X_test["actual"],
        numbered_X_test["predicted"]
    )
}

## for each student-instance in test, check how many of their actual courses are in the top 10 recommendations
## then find the mean of that number across all of the test students in this batch
student_instance_scores = []
for stu_instance, value in pivoted_dictionary.items():

    current_term_actual = value["current_term_actual"]
    current_term_prediction = value["current_term_prediction"]

    scores = []
    for term in current_term_actual:
        scores.append(int(term in current_term_prediction))
        
    student_instance_scores.append(np.mean(scores))

pct_correct_in_top_10 = np.mean(student_instance_scores)

print(pct_correct_in_top_10)



