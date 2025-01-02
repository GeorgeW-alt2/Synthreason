#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <cassert>
#include <random>
#include <string>

using namespace std;

int KB_LIMIT = 750;
int GEN_LEN = 140;
int EPOCHS = 5;
// Sigmoid activation function
double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of sigmoid function
double sigmoid_derivative(double x)
{
    return x * (1.0 - x);
}

// Neural Network Class
class NeuralNetwork
{
public:
    // Number of nodes in each layer
    int input_size, hidden_size, output_size;

    // Weights and biases
    vector<vector<double>> weights_input_hidden, weights_hidden_output;
    vector<double> bias_hidden, bias_output;

    // Constructor to initialize the network
    NeuralNetwork(int input, int hidden, int output)
    {
        input_size = input;
        hidden_size = hidden;
        output_size = output;

        // Use random number generation for weights and biases initialization
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(-0.5, 0.5);

        weights_input_hidden = random_matrix(input_size, hidden_size, gen, dis);
        weights_hidden_output = random_matrix(hidden_size, output_size, gen, dis);
        bias_hidden = random_vector(hidden_size, gen, dis);
        bias_output = random_vector(output_size, gen, dis);
    }

    // Random matrix initialization
    vector<vector<double>> random_matrix(int rows, int cols, mt19937& gen, uniform_real_distribution<>& dis)
    {
        vector<vector<double>> matrix(rows, vector<double>(cols));
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                matrix[i][j] = dis(gen);
            }
        }
        return matrix;
    }

    // Random vector initialization
    vector<double> random_vector(int size, mt19937& gen, uniform_real_distribution<>& dis)
    {
        vector<double> vec(size);
        for (int i = 0; i < size; ++i)
        {
            vec[i] = dis(gen);
        }
        return vec;
    }

    // Feedforward function to calculate the output
    vector<double> feedforward(const vector<double>& inputs)
    {
        vector<double> hidden_output(hidden_size);
        vector<double> final_output(output_size);

        // Hidden layer calculations
        for (int h = 0; h < hidden_size; ++h)
        {
            hidden_output[h] = bias_hidden[h];
            for (int i = 0; i < input_size; ++i)
            {
                hidden_output[h] += inputs[i] * weights_input_hidden[i][h];
            }
            hidden_output[h] = sigmoid(hidden_output[h]);
        }

        // Output layer calculations
        for (int o = 0; o < output_size; ++o)
        {
            final_output[o] = bias_output[o];
            for (int h = 0; h < hidden_size; ++h)
            {
                final_output[o] += hidden_output[h] * weights_hidden_output[h][o];
            }
            final_output[o] = sigmoid(final_output[o]);
        }

        return final_output;
    }

    // Training using backpropagation
    void train(const vector<vector<double>>& inputs, const vector<vector<double>>& targets, int epochs, double learning_rate)
    {
        vector<double> hidden_output(hidden_size);
        vector<double> output(output_size);
        vector<double> output_error(output_size);
        vector<double> output_delta(output_size);
        vector<double> hidden_error(hidden_size);
        vector<double> hidden_delta(hidden_size);

        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            double loss = 0;
            for (size_t i = 0; i < inputs.size()-1; ++i)
            {
                const vector<double>& input = inputs[i];
                const vector<double>& target = targets[i+1];

                // Feedforward
                for (int h = 0; h < hidden_size; ++h)
                {
                    hidden_output[h] = bias_hidden[h];
                    for (int j = 0; j < input_size; ++j)
                    {
                        hidden_output[h] += input[j] * weights_input_hidden[j][h];
                    }
                    hidden_output[h] = sigmoid(hidden_output[h]);
                }

                for (int o = 0; o < output_size; ++o)
                {
                    output[o] = bias_output[o];
                    for (int h = 0; h < hidden_size; ++h)
                    {
                        output[o] += hidden_output[h] * weights_hidden_output[h][o];
                    }
                    output[o] = sigmoid(output[o]);
                }

                // Calculate error and delta for output layer
                for (int o = 0; o < output_size; ++o)
                {
                    output_error[o] = target[o] - output[o];
                    output_delta[o] = output_error[o] * sigmoid_derivative(output[o]);
                }

                // Backpropagate to hidden layer
                for (int h = 0; h < hidden_size; ++h)
                {
                    hidden_error[h] = 0;
                    for (int o = 0; o < output_size; ++o)
                    {
                        hidden_error[h] += output_delta[o] * weights_hidden_output[h][o];
                    }
                    hidden_delta[h] = hidden_error[h] * sigmoid_derivative(hidden_output[h]);
                }

                // Update weights and biases
                for (int o = 0; o < output_size; ++o)
                {
                    for (int h = 0; h < hidden_size; ++h)
                    {
                        weights_hidden_output[h][o] += learning_rate * output_delta[o] * hidden_output[h];
                    }
                    bias_output[o] += learning_rate * output_delta[o];
                }

                for (int h = 0; h < hidden_size; ++h)
                {
                    for (int i = 0; i < input_size; ++i)
                    {
                        weights_input_hidden[i][h] += learning_rate * hidden_delta[h] * input[i];
                    }
                    bias_hidden[h] += learning_rate * hidden_delta[h];
                }

                // Calculate loss for this training step
                for (int o = 0; o < output_size; ++o)
                {
                    loss += pow(target[o] - output[o], 2);
                }
            }

            // Output loss for this epoch
            cout << "Epoch " << epoch << ", Loss: " << loss / inputs.size() << endl;
        }
    }
};


string to_lowercase(const string& str)
{
    string lower_str = str;
    transform(lower_str.begin(), lower_str.end(), lower_str.begin(), ::tolower);
    return lower_str;
}

vector<string> split(const string& str, char delimiter)
{
    vector<string> result;
    stringstream ss(str);
    string token;
    vector<string> currentGroup;

    while (getline(ss, token, delimiter))
    {
        currentGroup.push_back(to_lowercase(token));  // Convert token to lowercase

        // Once we have 3 words, combine them into one string and add to result
        if (currentGroup.size() == 3)
        {
            result.push_back(currentGroup[0] + " " + currentGroup[1] + " " + currentGroup[2]);
            currentGroup.clear();  // Reset the current group
        }
    }

    // Handle the case where there are less than 3 words left at the end
    if (!currentGroup.empty())
    {
        result.push_back(currentGroup[0]);
        if (currentGroup.size() > 1) result.back() += " " + currentGroup[1];
        if (currentGroup.size() > 2) result.back() += " " + currentGroup[2];
    }

    return result;
}


// Function to read the text file and generate a vocabulary
unordered_map<int, string> create_vocabulary(const string& filename)
{
    unordered_map<int, string> vocab;
    unordered_map<string, int> word_to_index;
    ifstream file(filename);
    string word;
    int index = 0;

    if (!file.is_open())
    {
        cerr << "Could not open the file!" << endl;
        return vocab;
    }

    vector<string> currentGroup;
    while (file >> word)
    {
        currentGroup.push_back(to_lowercase(word));  // Convert token to lowercase

        // Once we have 3 words, combine them into one string and add to vocabulary
        if (currentGroup.size() == 3)
        {
            string group = currentGroup[0] + " " + currentGroup[1] + " " + currentGroup[2];
            if (word_to_index.find(group) == word_to_index.end())  // Add new group to vocabulary
            {
                word_to_index[group] = index;
                vocab[index] = group;
                index++;
                if (index >= KB_LIMIT)
                {
                    break;
                }
            }
            currentGroup.clear();  // Reset the current group
        }
    }

    // Handle any remaining words if they are less than 3 at the end of the file
    if (!currentGroup.empty())
    {
        string group = currentGroup[0];
        if (currentGroup.size() > 1) group += " " + currentGroup[1];
        if (currentGroup.size() > 2) group += " " + currentGroup[2];
        if (word_to_index.find(group) == word_to_index.end())
        {
            word_to_index[group] = index;
            vocab[index] = group;
        }
    }

    file.close();
    return vocab;
}

// Function to map integer indices to words
string index_to_word(int index, const unordered_map<int, string>& vocab)
{
    auto it = vocab.find(index);
    if (it != vocab.end())
    {
        return it->second;
    }
    else
    {
        return "<UNKNOWN>";
    }
}


// Function to apply temperature scaling to probabilities
std::vector<double> apply_temperature(const std::vector<double>& logits, double temperature)
{
    std::vector<double> scaled_probs(logits.size());
    double sum = 0.0;

    for (size_t i = 0; i < logits.size(); ++i)
    {
        scaled_probs[i] = exp(logits[i] / temperature);
        sum += scaled_probs[i];
    }

    for (size_t i = 0; i < scaled_probs.size(); ++i)
    {
        scaled_probs[i] /= sum;
    }

    return scaled_probs;
}

// Function to sample a word index from the probability distribution
int sample_from_distribution(const std::vector<double>& probabilities)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());
    return dist(gen);
}

// Updated function to prepare input sequences
vector<vector<double>> prepare_sequences(const vector<string>& words, int sequence_length, const unordered_map<string, int>& word_to_index, int vocab_size)
{
    vector<vector<double>> sequences;
    for (size_t i = 0; i < words.size() - sequence_length; ++i)
    {
        vector<double> sequence(vocab_size * sequence_length, 0.0);
        for (int j = 0; j < sequence_length; ++j)
        {
            if (word_to_index.find(words[i + j]) != word_to_index.end())
            {
                sequence[word_to_index.at(words[i + j]) + j * vocab_size] = 1.0;
            }
        }
        sequences.push_back(sequence);
    }
    return sequences;
}

// Update main function to handle word sequences
int main()
{
    const int sequence_length = 3; // Number of words in each sequence
    double learningRate = 0.01;
    string filename = "test.txt"; // Replace with your actual file path

    // Create vocabulary from text file
    unordered_map<int, string> vocab = create_vocabulary(filename);
    int vocab_size = vocab.size();
    int input_size = vocab_size * sequence_length;
    int hidden_size = 50;
    int output_size = vocab_size;

    // Create reverse mapping for vocabulary
    unordered_map<string, int> word_to_index;
    for (const auto& pair : vocab)
    {
        word_to_index[pair.second] = pair.first;
    }

    // Prepare inputs and targets for training
    vector<vector<double>> inputs, targets;

    ifstream file(filename);
    string word;
    vector<string> words;
    int line_count = 0;

    while (file >> word)
    {
        words.push_back(word);
        if (line_count >= KB_LIMIT)
        {
            break;
        }
        line_count++;
    }

    inputs = prepare_sequences(words, sequence_length, word_to_index, vocab_size);

  for (size_t i = sequence_length; i < words.size(); ++i)
{
    vector<double> target(vocab_size, 0.0);
    auto it = word_to_index.find(words[i]); // Use find to check existence
    if (it != word_to_index.end())
    {
        target[it->second] = 2.0; // Access safely using the iterator
    }

    targets.push_back(target);
}


    NeuralNetwork model(input_size, hidden_size, output_size);

    // Train the network
    model.train(inputs, targets, EPOCHS, learningRate);

    cout << "Enter a sequence of " << sequence_length << " words to predict the next word (or type 'exit' to quit): " << endl;

    string user_input;
    double temperature = 0.7; // Adjust temperature to control randomness
    while (true)
    {
        getline(cin, user_input);

        if (user_input == "exit") // Allow user to exit
            break;

        // Split the input into words
        vector<string> input_words = split(user_input, ' ');



        // Prepare input sequence for prediction
        vector<double> input(input_size, 0.0);
        for (int i = 0; i < sequence_length; ++i)
        {
            if (word_to_index.find(input_words[i]) != word_to_index.end())
            {
                input[word_to_index.at(input_words[i]) + i * vocab_size] = 1.0;
            }
            else
            {
                cout << "Word not found in vocabulary: " << input_words[i] << endl;
                break;
            }
        }

        for (int i = 0; i < GEN_LEN; i++)
        {
            // Get the model's prediction
            vector<double> output = model.feedforward(input);
// Smooth the probabilities (additive smoothing)
            for (double& prob : output)
            {
                prob += 1e-3; // Small constant for smoothing
            }

// Normalize the probabilities
            double sum = accumulate(output.begin(), output.end(), 0.0);
            for (double& prob : output)
            {
                prob /= sum;
            }
            // Apply temperature scaling
            vector<double> probabilities = apply_temperature(output, temperature);

            // Sample the next word
            int predicted_index = sample_from_distribution(probabilities);
            string word = index_to_word(predicted_index, vocab);

            cout << word << " ";

            // Shift the sequence and add the predicted word
            input[predicted_index + (sequence_length - 1)] =2.0;
        }
        cout << endl;

        cout << "Enter another sequence or type 'exit' to quit: ";
    }

    return 0;
}
