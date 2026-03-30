/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */

// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  // By default, Docusaurus generates a sidebar from the docs folder structure
  tutorialSidebar: [
    {
      type: 'doc',
      id: 'intro',
      label: 'Introduction',
    },
    {
      type: 'category',
      label: 'Theory & Architecture',
      collapsed: false,
      items: [
        {
          type: 'doc',
          id: 'maths',
          label: 'Math Primitives',
        },
        {
          type: 'doc',
          id: 'penalty_builder',
          label: 'Penalty Builder',
        },
      ],
    },
    {
      type: 'category',
      label: 'API Manual',
      collapsed: false,
      items: [
        {
          type: 'doc',
          id: 'api/vla_core',
          label: 'VLA Core API',
        },
        {
          type: 'doc',
          id: 'api/deltanet',
          label: 'DeltaNet Baseline',
        },
        {
          type: 'doc',
          id: 'api/linear_transformer',
          label: 'Linear Transformer',
        },
      ],
    },
    {
      type: 'doc',
      id: 'experiments',
      label: 'Experiments & Results',
    },
    {
      type: 'doc',
      id: 'running',
      label: 'Getting Started',
    },
  ],

  // But you can create a sidebar manually
  /*
  tutorialSidebar: [
    'intro',
    'hello',
    {
      type: 'category',
      label: 'Tutorial',
      items: ['tutorial-basics/create-a-document'],
    },
  ],
   */
};

export default sidebars;
