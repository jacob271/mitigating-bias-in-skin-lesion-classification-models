import Image from 'next/image'
import { Inter } from 'next/font/google'
import ModalImage from 'react-modal-image';

const inter = Inter({ subsets: ['latin'] })

export default function Home() {
  return (
      <main className={`${inter.className}`}>
        <div className="container max-w-3xl mx-auto px-4 py-8">

          <section className="mb-8">
            <h1 className="text-4xl font-bold mb-2">Mitigating Bias in Skin Lesion Classification Models Using Variational Autoencoders</h1>
            <p className="text-lg text-gray-500">
              On this page, you find a short summary of my bachelor thesis.
            </p>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-semibold mb-2">Abstract</h2>
            <p>
              Leveraging deep learning for early detection of skin cancer could help prevent deaths. Current skin lesion classification algorithms include biases and perform worse for patients with rarer skin features. An existing bias mitigation method automatically detects rare skin features in a dataset using a Variational Autoencoder and takes them into account when training a classifier. We propose an adaptation of this method that allows having multiple classes. We show that the adaptation is effective in experiment setups similar to those in previous research. Bias with respect to age and skin tone of the patient was successfully reduced by more than 45%, with a significance of p &lt; 0.0005. Further, we observe that using transfer learning diminishes the bias mitigation effects while providing decreased biases on its own. Lastly, we find that the method is not effective for a more complex multi-class skin lesion classification task. We discuss potential reasons and areas for future work.
            </p>
          </section>

          <section className="mb-8">
            <h2 className="text-2xl font-semibold mb-2">Bias Mitigation Method</h2>
            <p>
              We applied an adapted version of the bias mitigation method from Amini et. al to a skin lesion classification task.
            </p>
            <p>
              First, we train a Variational Autoencoder on images of different skin lesions.
              A Variational Autoencoder is a neural network that learns to encode images into a latent space and decode them back from the latent space.

              </p>
            <ModalImage
                small="/vae_architecture.svg"
                large="/vae_architecture.svg"
                alt="Image 1"
                imageBackgroundColor={"#ffffff"}
                hideDownload={true}
                hideZoom={true}
                className="w-500 h-300 cursor-pointer mb-2 mt-2"
            />
            <p>
              The latent space is a lower-dimensional representation of the images which we use to determine images with rare skin features.

              This is of particular interest because it allows us to detect rare skin features without any human intervention.

              Next, we train a classifier on the images of the dataset. Hereby, we sample images with rare skin features more often.
              In doing so, the classifier learns to classify images with rare skin features better. We used the popular ResNet18 architecture as a classifier.
            </p>
          </section>
          <section className="mb-8">
            <h2 className="text-2xl font-semibold mb-2">Comparison to Related Work</h2>
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
                small="/simple_binary_accuracies.png"
                large="/simple_binary_accuracies.png"
                alt="Image 1"
                imageBackgroundColor={"#ffffff"}
                hideDownload={true}
                hideZoom={true}
                className="w-500 h-300 cursor-pointer mb-2 mt-2"
            />

            <p>
              Additionally, we measured bias, which we define as the variance of weighted accuracies for the different classes of an attribute.
              We observe that we don&apos;t have much bias with respect to sex in the first place. This can be explained by the fact that the dataset is balanced with respect to this attribute.
              With bias mitigation applied, the amount of bias does not change significantly.
              For the attribute visible hair, we observe a slight decrease in bias. However, the decrease is not significant.
            </p>
            <ModalImage
                small="/simple_binary_biases.png"
                large="/simple_binary_biases.png"
                alt="Image 1"
                imageBackgroundColor={"#ffffff"}
                hideDownload={true}
                hideZoom={true}
                className="w-500 h-300 cursor-pointer mb-2 mt-2"
            />
            <p>
              The last two attributes age and skin tone are more interesting. We observe a significant decrease in bias for both attributes.

              This suggests, that the bias mitigation method is indeed able to automatically detect images from patient with a rare age group or a rare skin tone. Also, this information is successfully used to improve overall accuracy and reduce bias.

              Overall, we showed with this experiment that the bias mitigation method is effective in a setup similar to related work.
            </p>
          </section>
          <section>
            <h2 className="text-2xl font-semibold mb-2">Heading 2</h2>
            <p>Paragraph 2</p>
            <ModalImage
                small="/multi_biases.png"
                large="/multi_biases.png"
                alt="Image 1"
                imageBackgroundColor={"#ffffff"}
                hideDownload={true}
                hideZoom={true}
                className="w-500 h-300 cursor-pointer mb-2 mt-2"
            />
          </section>

          <main className="grid grid-cols-1 gap-8">


            <section>
              <h2 className="text-2xl font-semibold mb-2">Heading 3</h2>
              <p>Paragraph 3</p>
              <ModalImage
                  small="/simple_binary_biases.png"
                  large="/simple_binary_biases.png"
                  alt="Image 1"
                  imageBackgroundColor={"#ffffff"}
                  hideDownload={true}
                  hideZoom={true}
                  className="w-500 h-300 cursor-pointer mb-2 mt-2"
              />
            </section>
          </main>

          <aside className="mt-8 flex justify-center">
            <div className="sidebar">
              <a href="https://www.linkedin.com/in/jsjacobschaefer" target="_blank" rel="noopener noreferrer">
                <Image src="/linkedin-banner.jpg" alt="LinkedIn Banner" width={200} height={100} />
              </a>
              <a href="https://jacobschaefer.de" target="_blank" rel="noopener noreferrer">
                <Image src="/website-banner.jpg" alt="Website Banner" width={200} height={100} />
              </a>
            </div>
          </aside>
        </div>
      </main>
  );
};