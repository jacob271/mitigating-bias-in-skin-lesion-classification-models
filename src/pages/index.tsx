import { Inter } from 'next/font/google'
import ModalImage from 'react-modal-image';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faGlobe, faCodeBranch} from '@fortawesome/free-solid-svg-icons';
import { faLinkedin } from '@fortawesome/free-brands-svg-icons';
const inter = Inter({ subsets: ['latin'] })

export default function Home() {
  return (
      <main className={`${inter.className}`}>
        <div className="container max-w-3xl mx-auto px-4 py-8">

          <section className="mb-8">
            <h1 className="text-5xl font-bold mb-2">Mitigating Bias in Skin Lesion Classification Models Using Variational Autoencoders</h1>
              <p className="text-lg text-gray-500">
                  This page provides a summary of my bachelor thesis. Check out my <a href="https://jacobschaefer.de" className="text-black hover:text-blue-700"><FontAwesomeIcon icon={faGlobe} className="mr-1" />Website</a>, take a look at some other projects of mine on my <a href="https://github.com/jacob271" className="text-black hover:text-blue-700"><FontAwesomeIcon icon={faCodeBranch} className="mr-1" />GitHub Account</a>, or connect with me on <a href="https://www.linkedin.com/in/jsjacobschaefer/" className="text-black hover:text-blue-700"><FontAwesomeIcon icon={faLinkedin} className="mr-1" />LinkedIn</a>.
              </p>
          </section>

          <section className="mb-8">
            <h2 className="text-3xl font-semibold mb-4">Abstract</h2>
            <p>
              Leveraging deep learning for early detection of skin cancer could help prevent deaths. Current skin lesion classification algorithms include biases and perform worse for patients with rarer skin features. An existing bias mitigation method automatically detects rare skin features in a dataset using a Variational Autoencoder and takes them into account when training a classifier. We propose an adaptation of this method that allows having multiple classes. We show that the adaptation is effective in experiment setups similar to those in previous research. Bias with respect to age and skin tone of the patient was successfully reduced by more than 45%, with a significance of p &lt; 0.0005. Further, we observe that using transfer learning diminishes the bias mitigation effects while providing decreased biases on its own. Lastly, we find that the method is not effective for a more complex multi-class skin lesion classification task. We discuss potential reasons and areas for future work.
            </p>
          </section>

          <section className="mb-8">
            <h2 className="text-3xl font-semibold mb-4">Bias Mitigation Method</h2>
            <p>
              We applied an adapted version of the bias mitigation method from Amini et. al to a skin lesion classification task.
            </p>
            <p>
              First, we train a Variational Autoencoder on images of different skin lesions.
              A Variational Autoencoder is a neural network that learns to encode images into a latent space and decode them back from the latent space.

              </p>
            <ModalImage
                small="/mitigating-bias-in-skin-lesion-classification-models/vae_architecture.svg"
                large="/mitigating-bias-in-skin-lesion-classification-models/vae_architecture.svg"
                imageBackgroundColor={"#ffffff"}
                hideDownload={true}
                hideZoom={true}
                className="w-500 h-300 cursor-pointer mb-4 mt-4"
            />
            <p>
              The latent space is a lower-dimensional representation of the images which we use to determine images with rare skin features.

              This is of particular interest because it allows us to detect rare skin features without any human intervention.

              Next, we train a classifier on the images of the dataset. Hereby, we sample images with rare skin features more often.
              In doing so, the classifier learns to classify images with rare skin features better. We used the popular ResNet18 architecture as a classifier.
            </p>
          </section>
          <section className="mb-8">
            <h2 className="text-3xl font-semibold mb-4">Comparison to Related Work</h2>
            <p>
              We performed an experiment where we follow a similar experiment setup as Sauman Das.
              we consider a skin lesion classification task with two classes. To evaluate the bias mitigation effect, we trained the model twice, once with the bias mitigation method and once without.
              Then, we compare the accuracies and biases of the two models with respect to age and sex of the patient, as well as the attribute visible hair and skin tone.
            </p>
            <p>
              As shown in the diagram below, we observe that the bias mitigation method improves the overall weighted accuracy by almost 7% on average, while also improving the weighted accuracy for every single attribute.
  the weighted accuracy for every single attribute.
            </p>

            <ModalImage
                small="/mitigating-bias-in-skin-lesion-classification-models/simple_binary_accuracies.png"
                large="/mitigating-bias-in-skin-lesion-classification-models/simple_binary_accuracies.png"
                imageBackgroundColor={"#ffffff"}
                hideDownload={true}
                hideZoom={true}
                className="w-500 h-300 cursor-pointer mb-4 mt-4"
            />

            <p>
              Additionally, we measured bias, which we define as the variance of weighted accuracies for the different classes of an attribute.
              We observe that we don&apos;t have much bias with respect to sex in the first place. This can be explained by the fact that the dataset is balanced with respect to this attribute.
              With bias mitigation applied, the amount of bias does not change significantly.
              For the attribute visible hair, we observe a slight decrease in bias. However, the decrease is not significant.
            </p>
            <ModalImage
                small="/mitigating-bias-in-skin-lesion-classification-models/simple_binary_biases.png"
                large="/mitigating-bias-in-skin-lesion-classification-models/simple_binary_biases.png"
                imageBackgroundColor={"#ffffff"}
                hideDownload={true}
                hideZoom={true}
                className="w-500 h-300 cursor-pointer mb-4 mt-4"
            />
            <p>
              The last two attributes age and skin tone are more interesting. We observe a significant decrease in bias for both attributes.

              This suggests, that the bias mitigation method is indeed able to automatically detect images from patient with a rare age group or a rare skin tone. Also, this information is successfully used to improve overall accuracy and reduce bias.

              Overall, we showed with this experiment that the bias mitigation method is effective in a setup similar to related work.
            </p>
          </section>
          <section className="mb-8">
            <h2 className="text-3xl font-semibold mb-4">Effects of Using Transfer Learning</h2>
            <p>
                Transfer learning is used in many deep learning applications to improve the performance of a model.
                Thus, we wanted to find out how using transfer learning effects the bias mitigation method.
            </p>
            <p>
                To do so, we performed an experiment where we trained the classifier with transfer learning. Once, with additional bias mitigation and once without.
                As in the previous experiment, we evaluated weighted accuracies and biases.
            </p>
              <p>
                  As expected, using transfer learning improves the overall weighted accuracy in comparison to not using transfer learning.
              </p>
            <ModalImage
                small="/mitigating-bias-in-skin-lesion-classification-models/transfer_with_binary_accuracies.png"
                large="/mitigating-bias-in-skin-lesion-classification-models/transfer_with_binary_accuracies.png"
                imageBackgroundColor={"#ffffff"}
                hideDownload={true}
                hideZoom={true}
                className="w-500 h-300 cursor-pointer mb-4 mt-4"
            />
              <p>
                  On top of that, we observe that using transfer learning and bias mitigation improves the overall weighted accuracy even further.
              </p>
              <p>
                  When taking a look at the biases below, we observe that using transfer learning alone leads to a decrease in bias.
                  However, additionally applying bias mitigation no longer leads to a significant decrease in bias.
              </p>
            <ModalImage
                small="/mitigating-bias-in-skin-lesion-classification-models/transfer_with_binary_bias.png"
                large="/mitigating-bias-in-skin-lesion-classification-models/transfer_with_binary_bias.png"
                imageBackgroundColor={"#ffffff"}
                hideDownload={true}
                hideZoom={true}
                className="w-500 h-300 cursor-pointer mb-4 mt-4"
            />
              <p>
                  To conclude, we showed that using transfer learning leads to reduced bias and improved weighted accuracy.
                  However, the bias mitigation method is no longer as effective.
              </p>

          </section>
          <section className="mb-8">
            <h2 className="text-3xl font-semibold mb-4">Bias Mitigation for Complex Multi-Class classification task</h2>
            <p>
                Lastly, we applied the method to a more complex classification task with four classes.
                With this setup, the bias mitigation method was not able to improve the performance or reduce bias.
            </p>
              <p>
                  Potential reasons for that include that the dataset was to small in order to extract meaningful information from the latent space of a Variational Autoencoder.
                  Future work, could investigate if using a separate Variational Autoencoder for each class could improve the bias mitigation effect.
                  Another promising approach could be to perform thorough hyperparameter tuning.
              </p>
            <ModalImage
                small="/mitigating-bias-in-skin-lesion-classification-models/multi_biases.png"
                large="/mitigating-bias-in-skin-lesion-classification-models/multi_biases.png"
                imageBackgroundColor={"#ffffff"}
                hideDownload={true}
                hideZoom={true}
                className="w-500 h-300 cursor-pointer mb-4 mt-4"
            />
          </section>
        </div>
      </main>
  );
};