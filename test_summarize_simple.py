from summarize_simple import *

def test_summarize():
    sample_text = (
        "Artificial intelligence is transforming industries. "
        "It enables machines to perform tasks that typically require human intelligence. "
        "AI systems are used in healthcare, finance, and transportation. "
        "They are capable of analyzing vast amounts of data and providing actionable insights. "
        "However, AI comes with ethical challenges and concerns about job displacement. "
        "Governments and organizations must address these challenges proactively."
    )
    stopwords = ["is", "are", "and", "of", "to", "in", "the", "that", "it", "they"]

    # Test extract_sentences
    sentences = extract_sentences(sample_text)
    assert len(sentences) == 6, f"Expected 6 sentences, got {len(sentences)}"
    assert "AI systems are used in healthcare, finance, and transportation." in sentences, "Expected sentence not found."

    # Test tokenize
    tokens = tokenize("AI systems are used in healthcare, finance, and transportation.", stopwords)
    expected_tokens = ["syst", "used", "heal", "fina", "tran"]
    assert tokens == expected_tokens, f"Expected {expected_tokens}, got {tokens}"

    # Test get_sentences_sim
    s1 = tokenize("AI systems are used in healthcare, finance, and transportation.", stopwords)
    assert s1 == ['syst', 'used', 'heal', 'fina', 'tran']
    s2 = tokenize("They are capable of analyzing vast amounts of finansial data.", stopwords)
    assert s2 == ['capa', 'anal', 'vast', 'amou', 'fina', 'data']

    similarity = get_sentences_sim(s1, s2)
    assert similarity == 0.1, f"Expected 0.1, got {similarity}"

    # Test get_matr_sim
    tokenized_sentences = [tokenize(s, stopwords) for s in sentences]
    similarity_matrix = get_matr_sim(tokenized_sentences)
    assert len(similarity_matrix) == len(sentences), "Similarity matrix size mismatch"
    assert all(len(row) == len(sentences) for row in similarity_matrix), "Non-square similarity matrix"

    # test similarity_matrix
    tokenized_sentences = [tokenize(s, stopwords) for s in sentences[:3]]
    similarity_matrix = get_matr_sim(tokenized_sentences)
    expected_similarity_matrix = [[0, round(1 / 11, 2), round(1 / 8, 2)], [round(1 / 11, 2), 0, 0], [round(1 / 8, 2), 0, 0]]
    for i in range(3):
        for j in range(3):
            if i != j:
                assert round(similarity_matrix[i][j], 2) == round(
                    get_sentences_sim(tokenized_sentences[i], tokenized_sentences[j]), 2)
            else:
                assert similarity_matrix[i][j] == 0

            assert round(similarity_matrix[i][j], 2) == expected_similarity_matrix[i][j]
            assert similarity_matrix[i][j] >= 0, "Negative score found"
            assert similarity_matrix[i][j] <= 1, "Big score found"

    assert similarity_matrix == [[0, 0.09090909090909091, 0.125], [0.09090909090909091, 0, 0.0], [0.125, 0.0, 0]]

    # Test calc_scores
    scores = calc_scores(similarity_matrix)
    expected_scores = [2.0, 0.42, 0.58]
    assert len(scores) == 3, "Scores length mismatch"
    assert all(score >= 0 for score in scores), "Negative score found"
    for i in range(3):
        assert round(scores[i], 2) == round(expected_scores[i], 2)

    # Test get_top_n_indexes
    scores = [0.5, 0.8, 0.2, 0.9, 0.3]
    top_indexes = get_top_n_indexes(scores, 3)
    assert top_indexes == [0, 1, 3], f"Expected [0, 1, 3], got {top_indexes}"

    # Test extract_sum
    summary = extract_sum(sample_text, 2)
    assert len(summary) == 2, f"Expected 2 summary sentences, got {len(summary)}"
    assert "Artificial intelligence is transforming industries." in summary, "Expected sentence not in summary"

    # Test summarize
    summary = summarize(sample_text, first_n=1, last_n=1)
    print(summary)
    assert len(summary.split('\n')) == 2, f"Expected 4 summary sentence, got {len(summary)}"
    assert "Artificial intelligence is transforming industries." in summary, "Expected first sentence missing"
    assert "Governments and organizations must address these challenges proactively." in summary, "Expected last sentence missing"

    text = "The quick brown fox jumps over the lazy dog. The dog barks at the fox. The fox runs away. The dog barks again. The fox jumps again."
    summary = summarize(text, n=4, first_n=0, last_n=0)
    assert len(summary.split('\n')) == 4
    assert "The quick brown fox jumps over the lazy dog." in summary
    assert "The dog barks at the fox." in summary
    assert "The fox jumps again." in summary

    summary = summarize(text, n=5, first_n=0, last_n=0)
    assert len(summary.split('\n')) == 5

    summary = summarize(text, n=6, first_n=0, last_n=0)
    assert len(summary.split('\n')) == 5

    summary = summarize(text, n=10, first_n=0, last_n=0)
    assert len(summary.split('\n')) == 5

    summary = summarize(text, n=4, first_n=3, last_n=2)
    assert len(summary.split('\n')) == 5

    print("All tests passed!")

if __name__ == "__main__":
    test_summarize()
