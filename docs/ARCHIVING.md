# Archive a code release on Zenodo

A software DOI identifies an archived code release. The ACM paper DOI identifies the article. This repository does not yet have a software DOI. Archiving the code will preserve the chosen version; it will not supply the data, results, or missing analysis runners listed in the README.

## Before the release

Choose the commit to archive after reviewing the code and documentation. Check that `CITATION.cff` has the correct author, affiliation, repository URL, and software license. Add the actual version and release date when they are known. Add an ORCID only after the author provides it.

Keep the software DOI unset until Zenodo assigns or reserves one. Use `CITATION.cff` for the archive metadata. A separate `.zenodo.json` is unnecessary here; when both files exist, Zenodo uses `.zenodo.json` and ignores `CITATION.cff`.

## Archive through GitHub

This route requires a public repository. When the repository is ready for publication:

1. Sign in to [Zenodo](https://zenodo.org/) with GitHub and authorize the connection.
2. Open your profile menu on Zenodo, select **GitHub**, and [enable](https://help.zenodo.org/docs/github/enable-repository/) `wilmerhenao/return-path-peer-retrieval`.
3. Add the actual version and release date to `CITATION.cff`, commit the changes, and tag that commit. Choose a release name, such as `v1.0.0`, only when ready to use it.
4. Publish a GitHub release for that tag. Once enabled, the integration archives each new published release and assigns a DOI. Ordinary commits and pushes do not create release DOIs.
5. Inspect the Zenodo record. Check its file list, author, title, license, version, and repository link. Record the tag and commit alongside the DOI.
6. Add the assigned software DOI to `CITATION.cff` and the README. Use the DOI for that specific version in the camera-ready paper. Add the ACM paper DOI to the paper citation when ACM assigns it.

The DOI is assigned after the release is archived, so the first archived commit cannot already contain that DOI through this route. Add it to the repository afterward and keep the archived tag pointing to the original release commit.

## Reserve a DOI before public release

Zenodo's GitHub integration cannot reserve a DOI before a release. A manual Zenodo upload can:

1. Create a new Zenodo upload draft and describe it as software. Enter and check the author, title, version, license, and repository URL using `CITATION.cff` as a reference; the manual upload does not import them through the GitHub integration.
2. Reserve a DOI in the draft's DOI field.
3. Prepare a ZIP archive of the chosen source commit. Record its commit and checksum, and describe the release's omissions in the record.
4. Review the files and metadata before publishing the record. A reserved DOI does not resolve to a published record until publication.

If using this manual route, use that record's versioning for later releases. Do not also archive the same release through the GitHub integration and create duplicate records. Confirm the final release files and public-access plan before publishing.

## Which DOI to cite

Zenodo assigns a **version DOI** for a specific release and a **concept DOI** covering its versions. Cite the version DOI in the paper so readers can identify the exact code snapshot. The concept DOI can be used when referring to the project across releases.

## Documentation

- [GitHub: archive a repository with Zenodo](https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content)
- [Zenodo: describe software with CITATION.cff](https://help.zenodo.org/docs/github/describe-software/citation-file/)
- [Zenodo: metadata precedence](https://help.zenodo.org/docs/github/describe-software/zenodo-json/)
- [Zenodo: reserve a DOI](https://help.zenodo.org/docs/deposit/describe-records/reserve-doi/)
- [Zenodo: DOI reservation and GitHub](https://support.zenodo.org/help/en-gb/24-github-integration/73-can-i-pre-reserved-a-doi-before-a-github-release)
- [Zenodo: version and concept DOIs](https://support.zenodo.org/help/en-gb/1-upload-deposit/97-what-is-doi-versioning)
