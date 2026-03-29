import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--primary button--lg"
            to="/docs/maths">
            Explore the Mathematics 🧠
          </Link>
          <Link
            className="button button--secondary button--lg"
            style={{marginLeft: '15px'}}
            to="https://github.com/deepbrain-labs/variational-linear-attention">
            GitHub Repository
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title}`}
      description="Documentation for Variational Linear Attention by DeepBrain Labs">
      <HomepageHeader />
      <main>
        <section className={styles.features}>
          <div className="container">
            <div className="row">
              <div className={clsx('col col--4')}>
                <div className="text--center padding-horiz--md feature-card">
                  <Heading as="h3">Mathematically Rigorous</Heading>
                  <p>Implementations built natively on optimal updating rules and carefully analyzed stability primitives, ensuring absolute numerical precision.</p>
                </div>
              </div>
              <div className={clsx('col col--4')}>
                <div className="text--center padding-horiz--md feature-card">
                  <Heading as="h3">Highly Scalable</Heading>
                  <p>True $O(N)$ linear complexity attention mechanisms, extensively optimized to gracefully handle extreme sequence lengths without memory collapse.</p>
                </div>
              </div>
              <div className={clsx('col col--4')}>
                <div className="text--center padding-horiz--md feature-card">
                  <Heading as="h3">DeepBrain Labs</Heading>
                  <p>Pioneering the next generation of foundational sequence models, unlocking unprecedented capabilities in reasoning and long-context understanding.</p>
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}
