//
// Created by nhatminh2947 on 11/18/19.
//

#include <GaussianDataGenerator.h>
#include <Point.h>
#include <IdxFileReader.h>
#include <ConfusionMatrix.h>


int N_DATA = 60000;

void E_step(Matrix<double> images, Matrix<double> mu, Matrix<double> pi, Matrix<double> &z) {
    double p_label[10];

    for (double &i : p_label) {
        i = 0;
    }

    for (int image_id = 0; image_id < N_DATA; ++image_id) {
        for (int label = 0; label < 10; ++label) {
            double p_image = 1.0;

            for (int pixel_id = 0; pixel_id < 28 * 28; ++pixel_id) {
                p_image *= ((images(pixel_id, image_id) == 1) ? mu(pixel_id, label) : (1 - mu(pixel_id, label)));
            }
            p_label[label] = pi(label, 0) * p_image;
        }

        double marginal = 0;
        for (double k : p_label) {
            marginal += k;
        }

        if (marginal == 0) {
            marginal = 1;
        }

        for (int k = 0; k < 10; ++k) {
            z(k, image_id) = p_label[k] / marginal;
        }
    }
}

void M_step(Matrix<double> images, Matrix<double> &mu, Matrix<double> &pi, Matrix<double> z) {
    double N[10];
    for (double &i : N) {
        i = 0;
    }

    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < N_DATA; ++j) {
            N[i] += z(i, j);
        }
    }

    for (int pixel_id = 0; pixel_id < 28 * 28; pixel_id++) {
        for (int label = 0; label < 10; label++) {
            double mu_pixel_label = 0.0;
            for (int image_id = 0; image_id < N_DATA; image_id++) {
                mu_pixel_label += images(pixel_id, image_id) * z(label, image_id);
            }

            mu(pixel_id, label) = mu_pixel_label / ((N[label] == 0) ? 1 : N[label]);
        }
    }

    for (int label = 0; label < 10; label++) {
        pi(label, 0) = N[label] / N_DATA;
    }
}


Matrix<double> ExtractFeatures(Matrix<double> &images) {
    Matrix<double> result(images.getRows(), images.getCols());
    int threshold = 255 / 2;

    for (int i = 0; i < 28 * 28; ++i) {
        for (int j = 0; j < N_DATA; ++j) {
            result(i, j) = int(images(i, j)) / threshold;
        }
    }

    return result;
}

double difference(Matrix<double> mu, const Matrix<double> &prev_mu) {
    Matrix<double> diff = mu - prev_mu;
    double total_diff = 0;

    for (int i = 0; i < 28 * 28; ++i) {
        for (int j = 0; j < 10; ++j) {
            total_diff += diff(i, j);
        }
    }
    return std::fabs(total_diff);
}

double sum_Pi(Matrix<double> pi) {
    double total = 0;
    for (int i = 0; i < 10; ++i) {
        total += pi(i, 0);
    }

    return total;
}

void reset(Matrix<double> &pi, Matrix<double> &mu, Matrix<double> &prev_mu, Matrix<double> &z) {
    pi = Matrix<double>(10, 1);
    mu = Matrix<double>(28 * 28, 10, fill::rand);
    prev_mu = Matrix<double>(28 * 28, 10, fill::zeros);
    z = Matrix<double>(10, N_DATA);

    for (int i = 0; i < 10; ++i) {
        pi(i, 0) = 0.1;
        for (int j = 0; j < N_DATA; ++j) {
            z(i, j) = 0.1;
        }
    }

    std::random_device r;
    std::default_random_engine generator{r()};
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (int i = 0; i < 28 * 28; ++i) {
        for (int j = 0; j < 10; ++j) {
            mu(i, j) = distribution(generator);
        }
    }
}

void condition_check(Matrix<double> &pi, Matrix<double> &mu, Matrix<double> &prev_mu, Matrix<double> &z, int &count) {
    double temp = 0;

    for (int i = 0; i < 10; ++i) {
        if (pi(i, 0) == 0) {
            count = 0;
            temp = 1;

            pi = Matrix<double>(10, 1);
            mu = Matrix<double>(28 * 28, 10, fill::rand);
            z = Matrix<double>(10, N_DATA);

            for (int class_id = 0; class_id < 10; ++class_id) {
                pi(class_id, 0) = 0.1;
                for (int image_id = 0; image_id < N_DATA; ++image_id) {
                    z(class_id, image_id) = 0.1;
                }
            }

            std::random_device r;
            std::default_random_engine generator{r()};
            std::uniform_real_distribution<double> distribution(0.0, 1.0);

            for (int pixel_id = 0; pixel_id < 28 * 28; ++pixel_id) {
                for (int class_id = 0; class_id < 10; ++class_id) {
                    mu(pixel_id, class_id) = distribution(generator);
                }
            }

            break;
        }
    }

    if (temp == 0) {
        count += 1;
    }
}

void print_mu(Matrix<double> mu) {
    mu = mu.T();

    for (int i = 0; i < 10; ++i) {
        std::cout << "class " << i << ":" << std::endl;
        for (int j = 0; j < 28; ++j) {
            for (int k = 0; k < 28; ++k) {
                std::cout << ((mu(i, j * 28 + k) >= 0.5) ? 1 : 0) << " ";
            }
            std::cout << std::endl;
        }
    }
}

std::vector<int> predict(Matrix<double> images, Matrix<double> mu, Matrix<double> pi) {
    std::vector<int> prediction;
    for (int n = 0; n < N_DATA; ++n) {
        double probs[10];
        for (double &prob : probs) {
            prob = 0;
        }

        for (int k = 0; k < 10; ++k) {
            double temp1 = 1;

            for (int pixel_id = 0; pixel_id < 28 * 28; ++pixel_id) {
                temp1 *= (images(pixel_id, n) ? mu(pixel_id, k) : (1 - mu(pixel_id, k)));
            }

            probs[k] = pi(k, 0) * temp1;
        }

        double max_value = -1;
        int c = -1;
        for (int k = 0; k < 10; ++k) {
            if (max_value < probs[k]) {
                max_value = probs[k];
                c = k;
            }
        }

        prediction.emplace_back(c);
    }

    return prediction;
}

std::vector<int> decide_label(std::vector<int> labels, std::vector<int> predicted) {
    std::vector<int> relation;
    for (int i = 0; i < 10; ++i) {
        relation.push_back(-1);
    }

    int count[10][10];

    for (auto &row : count) {
        for (int &element : row) {
            element = 0;
        }
    }

    for (int n = 0; n < N_DATA; ++n) {
        count[labels[n]][predicted[n]]++;
    }

    for (int times = 0; times < 10; ++times) {
        int label = 0;
        int cluster = 0;
        double max_value = INT64_MIN;
        for (int class_id = 0; class_id < 10; ++class_id) {
            for (int cluster_id = 0; cluster_id < 10; ++cluster_id) {
                if (max_value < count[class_id][cluster_id]) {
                    max_value = count[class_id][cluster_id];

                    label = class_id;
                    cluster = cluster_id;
                }
            }
        }

        relation[cluster] = label;

        for (int i = 0; i < 10; ++i) {
            count[i][cluster] = -1;
            count[label][i] = -1;
        }
    }

    return relation;
}

void hw2() {
    N_DATA = 60000;
//    Matrix<double> X = IdxFileReader::ReadImages("../src/HW2/t10k-images-idx3-ubyte");
//    std::vector<int> labels = IdxFileReader::ReadLabels("../src/HW2/t10k-labels-idx1-ubyte");
    Matrix<double> X = IdxFileReader::ReadImages("../src/HW2/train-images-idx3-ubyte", N_DATA);
    std::vector<int> labels = IdxFileReader::ReadLabels("../src/HW2/train-labels-idx1-ubyte", N_DATA);
    X = ExtractFeatures(X);
    Matrix<double> pi(10, 1);
    Matrix<double> mu(28 * 28, 10, fill::zeros);
    Matrix<double> prev_mu(28 * 28, 10, fill::zeros);
    Matrix<double> z(10, N_DATA);

    std::random_device r;
    std::default_random_engine generator{r()};
    std::uniform_real_distribution<double> distribution(0, 1);

    for (int i = 0; i < 28 * 28; ++i) {
        for (int j = 0; j < 10; ++j) {
            mu(i, j) = distribution(generator);
        }
    }

    for (int i = 0; i < 10; ++i) {
        pi(i, 0) = 0.1;
        for (int j = 0; j < N_DATA; ++j) {
            z(i, j) = 0.1;
        }
    }

    int iteration = 0;
    int count = 0;

    while (true) {
        iteration++;
        E_step(X, mu, pi, z);
        M_step(X, mu, pi, z);
        std::cout << pi << std::endl;
        condition_check(pi, mu, prev_mu, z, count);
        double diff = difference(mu, prev_mu);

        prev_mu = mu;

        print_mu(mu);
        std::cout << "No. of Iteration: " << iteration << ", Difference: " << diff << std::endl << std::endl;
        std::cout << "------------------------------------------------------------" << std::endl << std::endl;

        if (diff < 20 && count > 8 && sum_Pi(pi) >= 0.95) {
            break;
        }

        if(iteration == 10) {
            break;
        }
    }

    std::cout << "DONE==================" << std::endl;
    std::vector<int> predicted = predict(X, mu, pi);
    std::vector<int> relation = decide_label(labels, predicted);

    for (int n = 0; n < N_DATA; ++n) {
        predicted[n] = relation[predicted[n]];
    }
    std::cout << "DONE========AAAAAA====" << std::endl;

    for (int k = 0; k < 10; ++k) {
        std::vector<int> true_label_k = labels;
        std::vector<int> predicted_label_k = predicted;

        for (int & label : true_label_k) {
            label = (label == k) ? 1 : 0;
        }

        for (int & pred_label : predicted_label_k) {
            pred_label = (pred_label == k) ? 1 : 0;
        }

        ConfusionMatrix cm(true_label_k, predicted_label_k, 2);
        std::cout << cm << std::endl;
    }

    double error_rate = 0;
    for (int i = 0; i < N_DATA; ++i) {
        error_rate += (predicted[i] != labels[i]);
    }

    std::cout << "Total iteration to converge: " << iteration << std::endl;
    std::cout << "Total error rate: " << error_rate / N_DATA << std::endl;
}

int main(int argc, const char *argv[]) {
    hw2();
    return 0;
}
