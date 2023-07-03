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

          <section>
            <h2 className="text-2xl font-semibold mb-2">Abstract</h2>
            <p>
              Leveraging deep learning for early detection of skin cancer could help prevent deaths. Current skin lesion classification algorithms include biases and perform worse for patients with rarer skin features. An existing bias mitigation method automatically detects rare skin features in a dataset using a Variational Autoencoder and takes them into account when training a classifier. We propose an adaptation of this method that allows having multiple classes. We show that the adaptation is effective in experiment setups similar to those in previous research. Bias with respect to age and skin tone of the patient was successfully reduced by more than 45%, with a significance of p &lt; 0.0005. Further, we observe that using transfer learning diminishes the bias mitigation effects while providing decreased biases on its own. Lastly, we find that the method is not effective for a more complex multi-class skin lesion classification task. We discuss potential reasons and areas for future work.
            </p>
          </section>

          <main className="grid grid-cols-1 gap-8">
            <section>
              <h2 className="text-2xl font-semibold mb-2">Results</h2>
              <p>Paragraph 1</p>
              <ModalImage
                  small="/multi_accuracies.png"
                  large="/multi_accuracies.png"
                  alt="Image 1"
                  imageBackgroundColor={"#ffffff"}
                  hideDownload={true}
                  hideZoom={true}
                  className="w-500 h-300 cursor-pointer"
              />
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
                  className="w-500 h-300 cursor-pointer"
              />
            </section>

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
                  className="w-500 h-300 cursor-pointer"
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