def create_strat_key(row):
    # Combine relevant features to form a group key
    # Exclude intensity for neutral since there's no strong version
    if row['emotion'] == 1:
        return f"{row['emotion']}_{row['statement']}_{row['repetition']}"
    else:
        return f"{row['emotion']}_{row['intensity']}_{row['statement']}_{row['repetition']}"

df['strat_key'] = df.apply(create_strat_key, axis=1)

from sklearn.model_selection import train_test_split

# 1. First split: Train+Val and Test (80% - 20%)
train_val_df, test_df = train_test_split(
    df,
    test_size=0.05,
    stratify=df['strat_key'],
    random_state=42
)

# 2. Second split: Train and Validation (e.g., 80% of train_val for training)
train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.1,  # 0.2 * 0.9 = 0.18 → So final split = 72% train, 18% val, 10% test
    stratify=train_val_df['strat_key'],
    random_state=42
)

# ✅ Sanity check
print(f"Train size: {len(train_df)}")
print(f"Validation size: {len(val_df)}")
print(f"Test size: {len(test_df)}")

def plot_distribution(df, subset_name, column, color=None):
    fig = px.histogram(df, x=column, color=color or column,
                       title=f"{column.capitalize()} Distribution in {subset_name}",
                       barmode='group', template='plotly_dark')
    fig.show()

import plotly.express as px


# Emotion distribution
plot_distribution(train_df, "Train Set", "emotion_label")
plot_distribution(val_df, "Validation Set", "emotion_label")
plot_distribution(test_df, "Test Set", "emotion_label")

# Intensity (excluding neutral)
plot_distribution(train_df[train_df['emotion'] != 1], "Train Set (non-neutral)", "intensity_label", color="emotion_label")
plot_distribution(val_df[val_df['emotion'] != 1], "Validation Set (non-neutral)", "intensity_label", color="emotion_label")
plot_distribution(test_df[test_df['emotion'] != 1], "Test Set (non-neutral)", "intensity_label", color="emotion_label")

# Gender vs Emotion
plot_distribution(train_df, "Train Set", "emotion_label", color="gender_label")
plot_distribution(val_df, "Validation Set", "emotion_label", color="gender_label")
plot_distribution(test_df, "Test Set", "emotion_label", color="gender_label")

# !pip install sweetviz

import sweetviz as sv

# Compare train and validation
report1 = sv.compare([train_df, "Train"], [val_df, "Validation"])
report1.show_html("sweetviz_train_vs_val.html")

# Compare train and test
report2 = sv.compare([train_df, "Train"], [test_df, "Test"])
report2.show_html("sweetviz_train_vs_test.html")

from ydata_profiling import ProfileReport

# Profiling for each set
ProfileReport(train_df, title="Train Set Report").to_file("train_profile.html")
ProfileReport(val_df, title="Validation Set Report").to_file("val_profile.html")
ProfileReport(test_df, title="Test Set Report").to_file("test_profile.html")

train_df['category'] = 'train'
test_df['category'] = 'test'
val_df['category'] = 'valid'

final_df = pd.concat([train_df, test_df, val_df], axis=0, ignore_index=True)

print(final_df.head())