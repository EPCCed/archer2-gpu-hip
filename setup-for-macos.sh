
# The following commands should be sufficient for installing Jekyll (a static site generator) on your MacBook.
# These have been taken from https://jekyllrb.com/docs/installation/macos/ .

brew install chruby ruby-install
ruby-install ruby 3.4.1

echo "source $(brew --prefix)/opt/chruby/share/chruby/chruby.sh" >> ~/.zshrc
echo "source $(brew --prefix)/opt/chruby/share/chruby/auto.sh" >> ~/.zshrc
echo "chruby ruby-3.4.1" >> ~/.zshrc

sudo gem install jekyll

# After restarting your Terminal application, you should find that "ruby -v" returns "3.4.1".



# Once Jekyll has been installed, the following commands add the necessary Jekyll files for browsing your Markdown web pages.
# These are taken from https://docs.github.com/en/pages/setting-up-a-github-pages-site-with-jekyll/creating-a-github-pages-site-with-jekyll

cd /path/to/local/git/repo

jekyll new --skip-bundle .

# Within the "Gemfile" file replace the line that starts 'gem "jekyll"' with
# 'gem "github-pages", "~> GITHUB-PAGES-VERSION", group: :jekyll_plugins'
# The GITHUB-PAGES-VERSION label itself needs to be replaced with the github-pages version listed at https://pages.github.com/versions/ .

echo "Gemfile.lock" >> ./.gitignore
echo ".DS_Store" >> ./.gitignore



# Now from the repo folder, run the following command and then browse to http://127.0.0.1:4000/ .

bundle exec jekyll serve --baseurl=""
