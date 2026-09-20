import type { OceanPreset } from '../types/api';

export const OCEAN_PRESETS: OceanPreset[] = [
  {
    name: 'Bay of Bengal Freshwater Plume',
    description: 'Freshwater river runoff capping the surface; deeper barrier layer and thermocline.',
    latitude: 18.5,
    longitude: 88.25,
    date: '2023-06-15',
    region: 'Bay of Bengal',
  },
  {
    name: 'Central Arabian Sea',
    description: 'High surface salinity water mass with moderate thermocline depth.',
    latitude: 18.0,
    longitude: 65.0,
    date: '2023-06-15',
    region: 'Arabian Sea',
  },
  {
    name: 'Somali Upwelling Zone',
    description: 'Intense summer wind-driven upwelling bringing cold subsurface waters toward surface.',
    latitude: 10.0,
    longitude: 53.0,
    date: '2023-07-20',
    region: 'Western Arabian Sea',
  },
  {
    name: 'Equatorial Warm Pool',
    description: 'Equatorial low-Coriolis thermal structure with high SST and deep mixed layer.',
    latitude: 6.0,
    longitude: 80.0,
    date: '2023-05-10',
    region: 'Equatorial Indian Ocean',
  },
];
